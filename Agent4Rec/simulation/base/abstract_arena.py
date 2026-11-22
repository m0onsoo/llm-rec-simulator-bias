from argparse import Namespace
from recommenders.data import Data
import json
import hashlib
import sys
import os
import re
import torch
import random
import pandas as pd
import types
import numpy as np
from pathlib import Path
from simulation.run_io import atomic_json

RECOMMENDERS_PATH = str(Path(__file__).resolve().parents[2] / "recommenders")
sys.path.append(RECOMMENDERS_PATH)
from util import DataIterator
from models.base.utils import *
sys.path.remove(RECOMMENDERS_PATH)

# Inference does not use the evaluator, but the model module imports it through
# AbstractRS.  Supplying its import-time symbol avoids loading an optional
# Cython evaluator built against a different cluster compiler runtime.
evaluator_stub = types.ModuleType("evaluator")
evaluator_stub.ProxyEvaluator = object
sys.modules.setdefault("evaluator", evaluator_stub)
cython_tools_stub = types.ModuleType("util.cython.tools")
cython_tools_stub.float_type = np.float32
cython_tools_stub.is_ndarray = lambda value, dtype=None: isinstance(value, np.ndarray)
sys.modules.setdefault("util.cython.tools", cython_tools_stub)

float_type = np.float32
is_ndarray = lambda value, dtype=None: isinstance(value, np.ndarray)

class abstract_arena:
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.dataset = args.dataset
        self.val_users = args.val_users
        self.val_ratio = args.val_ratio
        self.simulation_name = args.simulation_name
        self.device = torch.device(f"cuda:{args.cuda}")
        self.n_avatars = args.n_avatars
        self.modeltype = args.modeltype
        self.items_per_page = args.items_per_page
        self.execution_mode = args.execution_mode
        self.rec_gt = args.rec_gt
        self.model_path = "recommenders/weights/" + args.dataset + "/" + args.modeltype + "/" + args.model_path
        print("============================")
        print(self.model_path)

    def excute(self):
        """
        The whole process of the simulation
        """
        if getattr(self.args, "position_order", "original") == "random" and self.args.add_advert:
            raise ValueError("position permutation experiments do not support advertisements")
        if (
            getattr(self.args, "activity_persona_transform", "identity") != "identity"
            and self.args.activity_persona_mode != "included"
        ):
            raise ValueError("activity persona transforms require --activity_persona_mode included")
        self.load_saved_args(self.model_path)
        self.prepare_dir()
        self.prepare_run_manifest()
        self.load_data()
        self.initialize_all_avatars()
        if self.args.rankings_path:
            self.load_full_rankings(self.args.rankings_path)
        else:
            self.load_recommender()
            self.get_full_rankings()
        self.load_additional_info()
        if(self.val_users):
            self.validate_all_avatars()
        else:
            self.simulate_all_avatars()
            self.save_results()

    def load_full_rankings(self, rankings_path):
        path = Path(rankings_path).expanduser().resolve()
        rankings = np.load(path)
        expected = (self.n_avatars, self.data.n_items)
        if rankings.ndim != 2 or rankings.shape[0] < expected[0] or rankings.shape[1] != expected[1]:
            raise ValueError(
                f"Cached rankings at {path} have shape {rankings.shape}; expected at least {expected}"
            )
        self.full_rankings = rankings[:self.n_avatars].copy()
        output = Path(self.storage_base_path) / "rankings" / f"full_rankings_{self.n_avatars}.npy"
        np.save(output, self.full_rankings)
        print(f"loaded cached full rankings from {path}")

    def load_saved_args(self, model_path):
        """
        load the recommender args, which is saved when training the recommender
        """
        self.saved_args = Namespace()
        # If the path exists, read.
        if(os.path.exists(model_path + '/args.txt')):
            with open(model_path + '/args.txt', 'r') as f:
                self.saved_args.__dict__ = json.load(f)
        else:
            with open("recommenders/weights/default_args.txt", 'r') as f:
                self.saved_args.__dict__ = json.load(f)
        # View current directory.
        self.saved_args.data_path = 'datasets/' # Modify the table of contents.
        self.saved_args.dataset = self.dataset
        self.saved_args.cuda = self.args.cuda
        self.saved_args.modeltype = self.modeltype
        # self.saved_args.nodrop = self.args.nodrop
    
    def prepare_dir(self):
        # make dir
        def ensureDir(dir_path):
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
        self.storage_base_path = f"storage/{self.dataset}/{self.modeltype}/" + self.simulation_name
        ensureDir(self.storage_base_path)
        # ensureDir(self.storage_base_path + "/avatars")
        ensureDir(self.storage_base_path + "/running_logs")
        ensureDir(self.storage_base_path + "/rankings")
        # ensureDir(self.storage_base_path + "/new_train")
        if not self.args.resume and os.path.exists(self.storage_base_path + "/system_log.txt"):
            os.remove(self.storage_base_path + "/system_log.txt")

    def prepare_run_manifest(self):
        """Prevent a resume from mixing results produced by different settings."""
        ranking_path = None
        ranking_sha256 = None
        if self.args.rankings_path:
            resolved = Path(self.args.rankings_path).expanduser().resolve()
            ranking_path = str(resolved)
            digest = hashlib.sha256()
            with open(resolved, "rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            ranking_sha256 = digest.hexdigest()

        manifest = {
            "dataset": self.args.dataset,
            "modeltype": self.args.modeltype,
            "model_path": self.args.model_path,
            "rankings_path": ranking_path,
            "rankings_sha256": ranking_sha256,
            "n_avatars": self.args.n_avatars,
            "items_per_page": self.args.items_per_page,
            "max_pages": self.args.max_pages,
            "seed": self.args.seed,
            "llm_model": self.args.llm_model,
            "llm_base_url": self.args.llm_base_url,
            "reasoning_effort": self.args.reasoning_effort,
            "llm_temperature": self.args.llm_temperature,
            "activity_persona_mode": self.args.activity_persona_mode,
            "add_advert": self.args.add_advert,
            "display_advert": self.args.display_advert,
            "advert_type": self.args.advert_type,
        }
        # Do not alter legacy vanilla manifests or resume compatibility unless
        # permutation was explicitly requested.
        if getattr(self.args, "position_order", "original") != "original":
            manifest.update({
                "position_order": self.args.position_order,
                "permutation_seed": self.args.permutation_seed,
                "position_permutation_version": 1,
            })
        # Preserve vanilla manifests and old-run resume compatibility unless
        # the rating-hidden intervention was explicitly requested.
        if getattr(self.args, "item_rating_mode", "included") != "included":
            manifest["item_rating_mode"] = self.args.item_rating_mode
        if getattr(self.args, "activity_persona_transform", "identity") != "identity":
            manifest["activity_persona_transform"] = self.args.activity_persona_transform
        path = Path(self.storage_base_path) / "run_manifest.json"
        if self.args.resume and path.is_file():
            previous = json.loads(path.read_text())
            if previous != manifest:
                differing = sorted(
                    key for key in set(previous) | set(manifest)
                    if previous.get(key) != manifest.get(key)
                )
                raise ValueError(
                    "Resume configuration differs from the existing run: "
                    + ", ".join(differing)
                )
        else:
            atomic_json(path, manifest)


    def load_data(self):
        """
        load the data for simulation
        """
        sys.path.append(RECOMMENDERS_PATH)
        try:
            exec('from recommenders.models.'+ self.saved_args.modeltype + ' import ' + self.saved_args.modeltype + '_Data') # load special dataset
            print('from recommenders.models.'+ self.saved_args.modeltype + ' import ' + self.saved_args.modeltype + '_Data')
            self.data = eval(self.saved_args.modeltype + '_Data(self.saved_args)') 
        except:
            print("no special dataset")
            self.data = Data(self.saved_args) # load data from the path
            print("finish loading data")
        sys.path.remove(RECOMMENDERS_PATH)
        # import pickle
        # with open(f'datasets/{self.dataset}/simulation/movie_dict.pkl', 'rb') as f:
        #     self.movie_detail = pickle.load(f)
        self.movie_detail = pd.read_csv(f'datasets/{self.dataset}/simulation/movie_detail.csv')        

    def load_recommender(self):
        """
        load the recommender for simulation
        """
        sys.path.append(RECOMMENDERS_PATH)
        self.running_model = self.saved_args.modeltype
        exec('from recommenders.models.'+ self.saved_args.modeltype + ' import ' + self.running_model) # import the model first
        self.model = eval(self.running_model + '(self.saved_args, self.data)') # initialize the model with the graph
        self.model.cuda(self.device)
        print("finish generating recommender")
        sys.path.remove(RECOMMENDERS_PATH)

        # load the checkpoint
        def restore_checkpoint(model, checkpoint_dir, device):
            """
            If a checkpoint exists, restores the PyTorch model from the checkpoint.
            Returns the model and the current epoch.
            """
            cp_files = [file_ for file_ in os.listdir(checkpoint_dir)
                        if file_.startswith('epoch=') and file_.endswith('.checkpoint.pth.tar')]
            if not cp_files:
                print('No saved model parameters found')
            epoch_list = []
            regex = re.compile(r'\d+')
            for cp in cp_files:
                epoch_list.append([int(x) for x in regex.findall(cp)][0])
            loading_epoch = max(epoch_list)

            filename = os.path.join(checkpoint_dir,
                                    'epoch={}.checkpoint.pth.tar'.format(loading_epoch))
            # print("Loading from checkpoint {}?".format(filename))

            checkpoint = torch.load(filename, map_location = str(device))
            model.load_state_dict(checkpoint['state_dict'])
            print("=> Successfully restored checkpoint (trained for {} epochs)"
                    .format(checkpoint['epoch']))

            return model, loading_epoch

        if(self.args.modeltype != "Random" and self.args.modeltype != "Pop"):
            print("loading checkpoint")
            self.model, self.loading_epoch = restore_checkpoint(self.model, self.model_path, self.device) # restore the checkpoint
            self.model.eval()
        # self.model, self.loading_epoch = restore_checkpoint(self.model, self.model_path, self.device) # restore the checkpoint
    
    def get_full_rankings(self, filename = "full_rankings", batch_size = 512):
        """
        document the full rankings of the items,
        according to a specific cf model
        """
        # if(os.path.exists(self.storage_base_path + '/{}_{}.npy'.format(filename, self.n_avatars))):
        #     print("loading full rankings from storage")
        #     self.full_rankings = np.load(self.storage_base_path + '/{}_{}.npy'.format(filename, self.n_avatars))
        #     print("finish loading full rankings")
        #     print(type(self.full_rankings))
        # else:
        # dump_dict = merge_user_list([self.data.train_user_list,self.data.valid_user_list])
        print("nodrop?", self.data.nodrop)
        # @ Use valid data for simulation.
        if(self.data.nodrop):
            dump_dict = merge_user_list([self.data.train_nodrop_user_list, self.data.test_user_list])
        else:
            dump_dict = merge_user_list([self.data.train_user_list, self.data.test_user_list])
        # dump_dict = merge_user_list([self.data.train_user_list, self.data.test_user_list])
        score_matrix = np.zeros((len(self.simulated_avatars_id), self.data.n_items))
        simulated_avatars_iter = DataIterator(self.simulated_avatars_id, batch_size=batch_size, shuffle=False, drop_last=False)
        for batch_id, batch_users in tqdm(enumerate(simulated_avatars_iter)):
            ranking_score = self.model.predict(batch_users, None)  # (B,N)
            if not is_ndarray(ranking_score, float_type):
                ranking_score = np.array(ranking_score, dtype=float_type)
            # set the ranking scores of training items to -inf,
            # then the training items will be sorted at the end of the ranking list.
            
            for idx, user in enumerate(batch_users):
                dump_items = dump_dict[user]
                # dump_items = [ x for x in dump_items if not x in self.data.test_user_list[user] ]
                ranking_score[idx][dump_items] = -np.inf

                score_matrix[batch_id*batch_size+idx] = ranking_score[idx]
                
            print('finish recommend one batch', batch_id)
            # break
        
        print('finish generating score matrix')
        self.full_rankings = np.argsort(-score_matrix, axis=1)
        if(self.rec_gt):
            # for user in self.simulated_avatars_id:
            #     for idx, item in enumerate(self.data.train_user_list[user]):
            #         self.full_rankings[user][idx] = item
            gt_dict = pd.read_pickle('scripts/user_ground_truth.pkl')
            for user in self.simulated_avatars_id:
                for idx, item in enumerate(gt_dict[user]):
                    self.full_rankings[user][idx] = item
        np.save(self.storage_base_path + '/rankings/' + '/{}_{}.npy'.format(filename, self.n_avatars), self.full_rankings)

        print('finish get full rankings')
    
    def initialize_all_avatars(self):
        """
        initialize all avatars
        """
        # all_avatars = sorted(list(self.data.test_user_list.keys()))
        # self.simulated_avatars_id = all_avatars[:self.n_avatars]
        self.simulated_avatars_id = list(range(self.n_avatars))
        # print('simulated avatars', self.simulated_avatars_id)
        # self.simulated_avatars_id = sorted(random.sample(all_avatars, self.n_avatars))

    def page_generator(self):
        """
        generate one page items for one avatar
        """
        raise NotImplementedError

    def validate_all_avatars(self):
        """
        validate the users
        """
        raise NotImplementedError

    def simulate_all_avatars(self):
        """
        excute the simulation for all avatars
        """
        raise NotImplementedError
    
    def simulate_one_avatar(self):
        """
        excute the simulation for one avatar
        """
        raise NotImplementedError
    
    def save_results(self):
        """
        save the results of the simulation
        """
        raise NotImplementedError
    
    def load_additional_info(self):
        """
        load additional information for the simulation
        """
        pass
