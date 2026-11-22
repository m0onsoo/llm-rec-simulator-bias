import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    # Overall settings
    parser.add_argument('--simulation_name', type=str, default= 'Test',
                        help='The name of one trial of simulation.')
    parser.add_argument('--cuda', type=int, default=0,
                        help='Specify which gpu to use.')
    parser.add_argument('--seed', type=int, default=101,
                        help='Random seed.')
    parser.add_argument('--items_per_page', type=int, default=4,
                        help='Number of items per page.')
    parser.add_argument('--num_avatars', type=int, default=20,
                        help='Number of avatars for sandbox simulation.')
    parser.add_argument('--execution_mode', type=str, default= 'parallel',
                        choices=['serial', 'parallel'],
                        help='Specify execution mode: serial or parallel.')
    parser.add_argument('--llm_model', type=str, default='gpt-6-luna',
                        help='OpenAI-compatible chat model used by every avatar call.')
    parser.add_argument('--llm_base_url', type=str, default='https://api.openai.com/v1')
    parser.add_argument('--reasoning_effort', type=str, default='none')
    parser.add_argument('--llm_temperature', type=float, default=0.0)
    parser.add_argument('--max_api_retries', type=int, default=8)
    parser.add_argument('--max_workers', type=int, default=32,
                        help='Maximum concurrent avatars in parallel execution.')
    parser.add_argument('--avatar_batch_size', type=int, default=0,
                        help='Users scheduled per async batch; 0 uses max_workers.')
    parser.add_argument('--batch_pause_seconds', type=float, default=0.0,
                        help='Cooldown between async avatar batches.')
    parser.add_argument('--activity_persona_mode', type=str, default='included',
                        choices=['included', 'excluded'],
                        help='Whether the activity-frequency description is exposed to the LLM.')
    parser.add_argument('--activity_persona_transform', type=str, default='identity',
                        choices=['identity', 'swap_extremes'],
                        help='Optional prompt-only activity mapping; swap_extremes maps 1<->3 and keeps 2.')
    parser.add_argument('--item_rating_mode', type=str, default='included',
                        choices=['included', 'excluded'],
                        help='Whether the per-item historical mean rating is exposed to the LLM.')
    parser.add_argument('--exclude_item_rating', dest='item_rating_mode',
                        action='store_const', const='excluded',
                        help='Alias for --item_rating_mode excluded.')
    parser.add_argument('--rankings_path', type=str, default=None,
                        help='Optional cached full-ranking .npy; skips recommender checkpoint inference.')
    parser.add_argument('--resume', action='store_true',
                        help='Resume a run by preserving completed users and rerunning incomplete users.')
    parser.add_argument('--position_order', '--position-order', dest='position_order', type=str, default='original',
                        choices=['original', 'random'],
                        help='Order within each recommendation page. The default preserves vanilla behavior exactly.')
    parser.add_argument('--permutation_seed', '--permutation-seed', dest='permutation_seed', type=int, default=0,
                        help='Seed used only when --position_order=random; independent of the simulation seed.')

    # Only recommend ground truth
    parser.add_argument("--rec_gt", action="store_true",
                        help="whether to recommend ground truth")
    
    # Using wandb
    parser.add_argument("--use_wandb", action="store_true",
                        help="whether to use wandb")
    
    # Only validate the effectiveness of agents
    parser.add_argument("--val_users", action="store_true",
                        help="whether to validate users")
    parser.add_argument('--val_ratio', type=int, default=1,
                        help='Ratio of unobserved items vs ground truth for validation.')
    
    # Advertisement settings
    parser.add_argument("--add_advert", action="store_true",
                        help="whether to add advertisement")
    parser.add_argument("--display_advert", action="store_true",
                        help="whether to display advertisement")
    parser.add_argument('--advert_type', type=str, default='pop_high',
                        choices=['all', 'pop_high', 'pop_low', 'unpop_high', 'unpop_low'],
                        help='Specify advertisement type.')
    
    # Dataset settings
    parser.add_argument('--dataset', type=str, default='ml-1m',
                        help='Dataset to use.')

    # Avatar settings
    parser.add_argument('--n_avatars', type=int, default=3,
                        help='How many avatars to simulate.')
    parser.add_argument('--max_pages', type=int, default=1,
                        help='The maximum page number users would like to view')


    # Recommender settings
    parser.add_argument('--model_path', type=str, default= 'Saved',
                        help='Specify model save path.')
    parser.add_argument('--modeltype', type=str, default= 'LightGCN',
                        help='Specify model save path.')

    # others
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Learning rate.')
    parser.add_argument("--pred_norm", action="store_true",
                        help="pred_norm")

    args, _ = parser.parse_known_args()

    return args


