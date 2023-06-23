from argparse import Namespace
from src.args_project import parse_args

args: Namespace = parse_args('DeepAssetAllocationOnly', False, 3000, 256)

if __name__ == '__main__':
    from src.calc_weights_realtime import calc_fixed_horizon_allocations, calc_term_fund_allocations, save_results
    import pandas as pd
    date_today = pd.to_datetime('today').strftime("%Y-%m-%d")
    alphas_tactical_fixed_horizon, alphas_strategic_fixed_horizon, alphas_tactical_JV_fixed_horizon, alphas_strategic_JV_fixed_horizon, date_fixed_horizon, invest_horizon_fixed_horizon, data_t, data_tplus1, unconditional_mean = calc_fixed_horizon_allocations(args, invest_horizon=48)
    alphas_tactical_target_date, alphas_strategic_target_date, alphas_tactical_JV_target_date, alphas_strategic_JV_target_date, date_target_date, invest_horizon_target_date =  calc_term_fund_allocations(args, term_date='12-31-2025')
    save_results(args.results_dir_save, f'real_time_allocations_{date_today}',alphas_tactical_fixed_horizon, alphas_strategic_fixed_horizon, alphas_tactical_JV_fixed_horizon, alphas_strategic_JV_fixed_horizon, date_fixed_horizon, invest_horizon_fixed_horizon, alphas_tactical_target_date, alphas_strategic_target_date, alphas_tactical_JV_target_date, alphas_strategic_JV_target_date, date_target_date, invest_horizon_target_date, data_t, data_tplus1, unconditional_mean)
