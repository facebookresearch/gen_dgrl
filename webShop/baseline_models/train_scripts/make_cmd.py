import argparse
import json
import os


def generate_train_cmds(
    params, num_trials=1, start_index=0, newlines=False, 
    xpid_generator=None, xpid_prefix='', xvfb=False, algo_type='il',
    include_wandb_group=False,
    count_set=None):
    separator = ' \\\n' if newlines else ' '
    
    cmds = []

    if xpid_generator:
        params['xpid'] = xpid_generator(params, xpid_prefix, algo_type)
        if include_wandb_group:
            params['wandb_group'] = params['xpid']

    start_seed = params['seed']

    for t in range(num_trials):
        params['seed'] = start_seed + t + start_index

        cmd = [f'python train_choice_{algo_type}.py']

        trial_idx = t + start_index
        for k,v in params.items():
            if k == 'xpid':
                v = f'{v}_{trial_idx}'

                if count_set is not None:
                    count_set.add(v)

            cmd.append(f'--{k}={v}')

        cmd = separator.join(cmd)

        cmds.append(cmd)

    return cmds


def generate_all_params_for_grid(grid, defaults={}):
    
    def update_params_with_choices(prev_params, param, choices):
        updated_params = []
        for v in choices:
            for p in prev_params:
                updated = p.copy()
                updated[param] = v
                updated_params.append(updated)

        return updated_params

    all_params = [{}]
    for param, choices in grid.items():
        all_params = update_params_with_choices(all_params, param, choices)

    full_params = []
    for p in all_params:
        d = defaults.copy()
        d.update(p)
        full_params.append(d)

    return full_params


def parse_args():
    parser = argparse.ArgumentParser(description='Make commands')
    
    parser.add_argument(
        '--dir',
        type=str,
        default='train_scripts/grid_configs/',
        help='Path to directory with .json configs')

    parser.add_argument(
        '--json',
        type=str,
        default=None,
        help='Name of .json config for hyperparameter search-grid')

    parser.add_argument(
        '--num_trials',
        type=int,
        default=1,
        help='Name of .json config for hyperparameter search-grid')

    parser.add_argument(
        '--start_index',
        default=0,
        type=int,
        help='Starting trial index of xpid runs')

    parser.add_argument(
        '--count',
        action='store_true',
        help='Print number of generated commands at the end of output.')


    parser.add_argument(
        "--checkpoint",
        action='store_true',
        help='Whether to start from checkpoint'
    )

    parser.add_argument(
        "--wandb_base_url",
        type=str,
        default=None,
        help='wandb base url'
    )
    parser.add_argument(
        "--wandb_api_key",
        type=str,
        default=None,
        help='wandb api key'
    )
    parser.add_argument(
        '--wandb_project',
        type=str,
        default=None,
        help='wandb project name')

    parser.add_argument(
        '--include_wandb_group',
        action="store_true",
        help='Whether to include wandb group in cmds.')

    parser.add_argument(
        '--use_ucb',
        action="store_true",
        help='Whether to include ucb arguments.')

    parser.add_argument(
        '--xvfb',
        action="store_true",
        help='Whether to use xvfb.')
    parser.add_argument(
        '--algo_type',
        type=str,
        default='il',
        choices=['il', 'cql', 'bcq'],
        help='which algo to run')

    return parser.parse_args()


def xpid_from_params(p, prefix='', algo_type='il'):
        
    algo_prefix = algo_type
    
    agent_prefix = f"bs{p['per_device_train_batch_size']}_lr{p['learning_rate']}"
    
    if p['image']==1:
        agent_prefix += '_image'
        
    if algo_type == 'cql':
        agent_prefix += f"_a{p['cql_alpha']}"
    elif algo_type == 'bcq':
        agent_prefix += f"_a{p['bcq_alpha']}"
        
    if algo_type in ['cql', 'bcq']:
        agent_prefix += f"_g{p['gamma']}_tuf{p['target_update_freq']}_tau{p['target_model_tau']}"
        
    return f'{algo_prefix}_{agent_prefix}'

if __name__ == '__main__':
    args = parse_args()

    # Default parameters
    params = {
        'xpid': 'debug',
        
        'task_name': 'mrpc',
        'per_device_train_batch_size': 1,
        'per_device_eval_batch_size': 8,
        'learning_rate': 2e-5,

        'output_dir': './ckpt/web_click',
        'image': 0,
        
        'seed': 0
        
    }
    if args.algo_type != 'il':
        params.update({
            'gamma': 0.99,
            'target_update_freq': 100,
            'target_model_tau': 0.005
        })
        if args.algo_type == 'cql':
            params.update({
                'cql_alpha': 4.0
            })
        elif args.algo_type == 'bcq':
            params.update({
                'bcq_alpha': 0.1
            })
            

    json_filename = args.json
    if not json_filename.endswith('.json'):
        json_filename += '.json'

    grid_path = os.path.join(os.path.expandvars(os.path.expanduser(args.dir)), json_filename)
    config = json.load(open(grid_path))
    grid = config['grid']
    xpid_prefix = '' if 'xpid_prefix' not in config else config['xpid_prefix']

    if args.checkpoint:
        params['checkpoint'] = True

    if args.wandb_project:
        params['wandb_project'] = args.wandb_project

    if args.wandb_base_url:
        params['wandb_base_url'] = args.wandb_base_url
    if args.wandb_api_key:
        params['wandb_api_key'] = args.wandb_api_key
        


    # Generate all parameter combinations within grid, using defaults for fixed params
    all_params = generate_all_params_for_grid(grid, defaults=params)

    unique_xpids = None
    if args.count:
        unique_xpids = set()

    # Print all commands
    count = 0
    for p in all_params:
        cmds = generate_train_cmds(p,
            num_trials=args.num_trials, 
            start_index=args.start_index, 
            newlines=True, 
            xpid_generator=xpid_from_params, 
            xpid_prefix=xpid_prefix,
            include_wandb_group=args.include_wandb_group,
            xvfb=args.xvfb,
            algo_type=args.algo_type,
            count_set=unique_xpids)

        for c in cmds:
            print(c + '\n')
            count += 1

    if args.count:
        print(f'Generated {len(unique_xpids)} unique commands.')