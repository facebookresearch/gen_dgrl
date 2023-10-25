from offline.agents.bc import BehavioralCloning
from offline.agents.bcq import BCQ
from offline.agents.ddqn_cql import CQL
from offline.agents.iql import IQL
from offline.agents.dt import DecisionTransformer

def _create_agent(args, env, extra_config):
    agent_name = args.algo
    if agent_name == "bc":
        return BehavioralCloning(env.observation_space, env.action_space.n, args.lr, args.agent_model, args.hidden_size)
    if agent_name == "bcq":
        assert args.agent_model in ["bcq", "bcqresnetbase"]
        return BCQ(env.observation_space, 
                   env.action_space.n, 
                   args.lr, 
                   args.agent_model, 
                   args.hidden_size,
                   gamma=args.gamma,
                   target_update_freq=args.target_update_freq,
                   tau=args.tau,
                   eps_start=args.eps_start,
                   eps_end=args.eps_end,
                   eps_decay=args.eps_decay,
                   bcq_threshold=args.bcq_threshold,
                   perform_polyak_update=args.perform_polyak_update)
    elif agent_name == "cql":
        return CQL(env.observation_space, 
                   env.action_space.n, 
                   args.lr, 
                   args.agent_model, 
                   args.hidden_size,
                   gamma=args.gamma,
                   target_update_freq=args.target_update_freq,
                   tau=args.tau,
                   eps_start=args.eps_start,
                   eps_end=args.eps_end,
                   eps_decay=args.eps_decay,
                   cql_alpha=args.cql_alpha,
                   perform_polyak_update=args.perform_polyak_update)
    elif agent_name == "iql":
        return IQL(env.observation_space, 
                   env.action_space.n, 
                   args.lr, 
                   args.agent_model, 
                   args.hidden_size,
                   gamma=args.gamma,
                   target_update_freq=args.target_update_freq,
                   tau=args.tau,
                   eps_start=args.eps_start,
                   eps_end=args.eps_end,
                   eps_decay=args.eps_decay,
                   iql_temperature=args.iql_temperature,
                   iql_expectile=args.iql_expectile,
                   perform_polyak_update=args.perform_polyak_update)
    elif agent_name in ["dt", "bct"]:
        return DecisionTransformer(env.observation_space,
                                   env.action_space.n, 
                                    args.agent_model, 
                                    extra_config["train_data_vocab_size"],
                                    extra_config["train_data_block_size"],
                                    extra_config["max_timesteps"],
                                    args.dt_context_length,
                                    extra_config["dataset_size"],
                                    lr=args.lr)
    else:
        raise ValueError(f"Invalid agent name {agent_name}.")