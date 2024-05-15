import argparse


def get_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--gamma', type=float, default=0.999, help='the discount factor of RL')
    parse.add_argument('--seed', type=int, default=333, help='the random seeds')
    parse.add_argument('--num-workers', type=int, default=1, help='the number of workers to collect samples')
    parse.add_argument('--env-name', type=str, default='Walker2d-v2',
                       help='the environment name: HalfCheetah-v2, Hopper-v2, Walker2d-v2, Ant-v2, Swimmer-v2 and Humanoid-v2')
    parse.add_argument('--batch-size', type=int, default=64, help='the batch size of updating')
    parse.add_argument('--lr', type=float, default=1e-4, help='learning rate of the algorithm')
    parse.add_argument('--train_epoch', type=int, default=10, help='the epoch during training')
    parse.add_argument('--nsteps', type=int, default=2048, help='the steps to collect samples')
    parse.add_argument('--vloss-coef', type=float, default=1, help='the coefficient of value loss')
    parse.add_argument('--ent-coef', type=float, default=0.0, help='the entropy loss coefficient')
    parse.add_argument('--tau', type=float, default=0.95, help='gae coefficient')
    parse.add_argument('--cuda', action='store_true', help='use cuda do the training')
    parse.add_argument('--transition_model_type', type=str, default='probabilistic', help='the transition model')
    parse.add_argument('--encoder_feature_dim', type=int, default=64, help='encoder feature dimension')
    parse.add_argument('--epochs', type=int, default=int(1e6), help='the total epochs for training')
    parse.add_argument('--dist', type=str, default='gauss', help='the distributions for sampling actions')
    parse.add_argument('--eps', type=float, default=1e-5, help='param for adam optimizer')
    parse.add_argument('--clip', type=float, default=0.2, help='the ratio clip param')
    parse.add_argument('--save-dir', type=str, default='saved_models/', help='the folder to save models')
    parse.add_argument('--lr-decay', action='store_true', help='if using the learning rate decay during decay')
    parse.add_argument('--lr-in', type=float, default=1e-4, help='the lr for intrinsic module')
    parse.add_argument('--max-grad-norm', type=float, default=1, help='grad norm')
    parse.add_argument('--display-interval', type=int, default=10, help='the interval that display log information')
    parse.add_argument('--env-type', type=str, default='mujoco', help='the type of the environment')
    parse.add_argument('--metric-type', type=str, default='avd', help='reward shaping')
    parse.add_argument('--log-dir', type=str, default='logs', help='the folders to save the log files')
    parse.add_argument('--r-ext-coef', type=float, default=1, help='dont use the extrinsic reward if 0')
    parse.add_argument('--r-in-coef', type=float, default=1, help='dont use the intrinsic reward if 0')
    parse.add_argument('--reward-delay-freq', type=int, default=11, help='the reward delay')
    parse.add_argument('--alpha', type=float, default=0.5, help='ratio of new and old,0 old,1 new,0.5 half')
    parse.add_argument('--log-data-dir', type=str, default='log_data', help='the log data for save .pt file')

    args = parse.parse_args()

    return args
