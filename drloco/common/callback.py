from os import makedirs, remove, rename
import os
import numpy as np
from tqdm import tqdm
import wandb

from stable_baselines3 import PPO
from drloco.config import config as cfgl
from drloco.config import hypers as cfg
from drloco.common import utils
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.callbacks import BaseCallback

# define intervals/criteria for saving the model
# save everytime the agent achieved an additional 10% of the max possible return
MAX_RETURN = cfg.ep_dur_max * 1 * cfg.rew_scale
EP_RETURN_INCREMENT = 0.1 * MAX_RETURN
# 10% of max possible reward
MEAN_REW_INCREMENT = 0.1 * cfg.rew_scale

# define evaluation interval
EVAL_MORE_FREQUENT_THRES = 3.2e6
EVAL_INTERVAL_RARE = 400e3 if not cfgl.DEBUG else 10e3
EVAL_INTERVAL_FREQUENT = 200e3
EVAL_INTERVAL_MOST_FREQUENT = 100e3
EVAL_INTERVAL = EVAL_INTERVAL_RARE

class TrainingMonitor(BaseCallback):
    def __init__(self, verbose=0):
        super(TrainingMonitor, self).__init__(verbose)
        # to control how often to save the model
        self.times_surpassed_ep_return_threshold = 0
        self.times_surpassed_mean_reward_threshold = 0
        # control evaluation
        self.n_steps_after_eval = EVAL_INTERVAL
        self.n_saved_models = 0
        self.moved_distances = []
        self.mean_walked_distance = 0
        self.min_walked_distance = 0
        self.mean_episode_duration = 0
        self.min_episode_duration = 0
        self.mean_walking_speed = 0
        self.min_walking_speed = 0
        self.mean_reward_means = 0
        self.tot_reward = 0
        self.best_tot_reward = 0
        self.count_stable_walks = 0
        self.summary_score = 0
        self.has_reached_stable_walking = False
        # collect the frequency of failed walks during evaluation
        self.failed_eval_runs_indices = []
        # log data less frequently
        self.skip_n_steps = 100
        self.skipped_steps = 99

    def _on_training_start(self) -> None:
        self.env = self.training_env
        # setup and launch tensorboard
        self.tb = SummaryWriter(log_dir=cfg.save_path + 'tb_logs/PPO_1', filename_suffix='_OWN_LOGS')
        utils.autolaunch_tensorboard(cfg.save_path, just_print_instructions=True)

    def _on_training_end(self):
        # stop logging to TB by stopping the SummaryWriter()
        self.tb.close()

    def _on_step(self) -> bool:
        if cfgl.DEBUG and self.num_timesteps > cfgl.MAX_DEBUG_STEPS:
            raise SystemExit(f"Planned Exit after {cfgl.MAX_DEBUG_STEPS} due to Debugging mode!")

        # reset the collection of episode lengths after 1M steps
        # goal: see a distribution of ep lens of last 1M steps,
        # ... not of the whole training so far...
        if self.num_timesteps % 1e6 < 1000:
            self.env.set_attr('ep_lens', [])

        self.n_steps_after_eval += 1 * cfg.n_envs

        # skip n steps to reduce logging interval and speed up training
        # if self.skipped_steps < self.skip_n_steps:
        #     self.skipped_steps += 1
        #     return True

        global EVAL_INTERVAL

        eval_freq = 100000 // cfg.n_envs
        if self.n_calls % eval_freq == 0:
            walking_stably = self.eval_walking()
        # if self.n_steps_after_eval >= EVAL_INTERVAL and not cfgl.DEBUG:
        #     self.n_steps_after_eval = 0
        #     walking_stably = self.eval_walking()

        ep_len = self.get_mean('ep_len_smoothed')
        ep_ret = self.get_mean('ep_ret_smoothed')
        mean_rew = self.get_mean('mean_reward_smoothed')

        # avoid logging data during first episode
        # if ep_len < {5: 8, 400: 60, 200:30, 50:8, 100:15}[cfgl.CTRL_FREQ]:
        #     return True
        log_freq = 25000 // cfg.n_envs
        if self.n_calls % log_freq == 0:
            self.log_to_tb(mean_rew, ep_len, ep_ret)
        # do not save a model if its episode length was too short
        # if ep_len > 1500:
        #     self.save_model_if_good(mean_rew, ep_ret)

        # reset counter of skipped steps after data was logged
        self.skipped_steps = 0

        return True


    def get_mean(self, attribute_name):
        values = self.env.get_attr(attribute_name)
        mean = np.mean(values)
        return mean


    def log_scalar(self, tag, value):
        """Logs a scalar value to TensorBoard."""
        self.tb.add_scalar(tag, value, self.num_timesteps)


    def log_to_tb(self, mean_rew, ep_len, ep_ret):
        # get the current policy
        model = self.model

        moved_distance = self.get_mean('moved_distance')
        # mean_abs_torque_smoothed = self.get_mean('mean_abs_ep_torque_smoothed')

        self.log_scalar('_det_eval/1. Summary Score []', self.summary_score),
        self.log_scalar('_det_eval/4. mean eval distance [m]', self.mean_walked_distance),
        self.log_scalar('_det_eval/5. MIN eval distance [m]', self.min_walked_distance),
        self.log_scalar('_det_eval/3. mean step reward [%]', self.mean_reward_means),
        self.log_scalar('_det_eval/6. mean episode duration [%]', self.mean_episode_duration),
        self.log_scalar('_det_eval/7. mean walking speed [m/s]', self.mean_walking_speed),

        self.log_scalar('_train/1. moved distance [m]', moved_distance),
        self.log_scalar('_train/2. episode length [%] (smoothed 0.75)', ep_len/cfg.ep_dur_max),
        self.log_scalar('_train/3. step reward [] (smoothed 0.25)', (mean_rew-cfg.alive_bonus)/cfg.rew_scale),
        self.log_scalar('_train/4. episode return [%] (smoothed 0.75)', (ep_ret-ep_len*cfg.alive_bonus)/(cfg.ep_dur_max*cfg.rew_scale)),
        self.log_scalar('_train/5. episode return [] (smoothed 0.75)', ep_ret),
        self.log_scalar('_train/6. episode return w/o alive bonus [] (smoothed 0.75)', (ep_ret-ep_len*cfg.alive_bonus)),
        self.log_scalar('_train/7. episode length [m] (smoothed 0.75)', ep_len),

        # log reward components
        mean_ep_pos_rew = self.get_mean('mean_ep_pos_rew_smoothed')
        mean_ep_vel_rew = self.get_mean('mean_ep_vel_rew_smoothed')
        mean_ep_com_rew = self.get_mean('mean_ep_com_rew_smoothed')
        self.log_scalar(f'_rews/1. mean ep pos rew ({cfg.n_envs}envs, smoothed 0.9)',
                                  mean_ep_pos_rew),
        self.log_scalar(f'_rews/2. mean ep vel rew ({cfg.n_envs}envs, smoothed 0.9)',
                          mean_ep_vel_rew),
        self.log_scalar(f'_rews/3. mean ep com rew ({cfg.n_envs}envs, smoothed 0.9)',
                          mean_ep_com_rew),

        self.log_scalar(f"_rews/total_reward", self.tot_reward)

        ep_pos_rew = self.get_mean('ep_pos_rew_smoothed')
        ep_vel_rew = self.get_mean('ep_vel_rew_smoothed')
        ep_com_rew = self.get_mean('ep_com_rew_smoothed')
        self.log_scalar(f'_rews/ep pos rew', ep_pos_rew),
        self.log_scalar(f'_rews/ep vel rew', ep_vel_rew),
        self.log_scalar(f'_rews/ep com rew', ep_com_rew),


    def save_model_if_good(self, mean_rew, ep_ret):
        if cfgl.DEBUG: return
        def get_mio_timesteps():
            return int(self.num_timesteps/1e6)

        ep_ret_thres = 0.6 * MAX_RETURN \
                       + int(EP_RETURN_INCREMENT * (self.times_surpassed_ep_return_threshold + 1))
        if ep_ret > ep_ret_thres:
            utils.save_model(self.model, cfg.save_path,
                             'ep_ret' + str(ep_ret_thres) + f'_{get_mio_timesteps()}M')
            self.times_surpassed_ep_return_threshold += 1
            print(f'NOT Saving model after surpassing EPISODE RETURN of {ep_ret_thres}.')
            # print('Model Path: ', cfg.save_path)

        # normalize reward
        mean_rew = (mean_rew - cfg.alive_bonus)/cfg.rew_scale
        mean_rew_thres = 0.4  \
                         + MEAN_REW_INCREMENT * (self.times_surpassed_mean_reward_threshold + 1)
        if mean_rew > (mean_rew_thres):
            # utils.save_model(self.model, cfg.save_path,
            #                  'mean_rew' + str(int(100*mean_rew_thres)) + f'_{get_mio_timesteps()}M')
            self.times_surpassed_mean_reward_threshold += 1
            print(f'NOT Saving model after surpassing MEAN REWARD of {mean_rew_thres}.')
            print('Model Path: ', cfg.save_path)


    def eval_walking(self):
        """
        Test the deterministic version of the current model:
        How far does it walk (in average and at least) without falling?
        @returns: If the training can be stopped as stable walking was achieved.
        """
        moved_distances, mean_rewards, ep_durs, mean_com_x_vels, tot_rew = [], [], [], [], []
        # save current model and environment
        checkpoint = f'{int(self.num_timesteps/1e5)}'
        model_path, env_path = \
            utils.save_model(self.model, cfg.save_path, checkpoint, full=False)

        # load the evaluation environment
        eval_env = utils.load_env(checkpoint, cfg.save_path, cfgl.ENV_ID)
        mimic_env = eval_env.venv.envs[0].env
        mimic_env.activate_evaluation()

        # load the saved model with the evaluation environment
        eval_model = PPO.load(model_path)

        # evaluate deterministically
        utils.log(f'Starting model evaluation, checkpoint {checkpoint}')
        obs = eval_env.reset()
        eval_n_times = 5
        video_freq = 500000 // cfg.n_envs
        should_record_video = self.n_calls % video_freq == 0
        for i in tqdm(range(eval_n_times)):
            ep_dur = 0
            walked_distance = 0
            rewards = []
            imgs = []
            while True:
                ep_dur += 1
                action, _ = eval_model.predict(obs, deterministic=True)
                obs, reward, done, info = eval_env.step(action)

                if i == eval_n_times - 1 and should_record_video:
                    img = eval_env.render("rgb_array")
                    imgs.append(img)
                if done:
                    moved_distances.append(walked_distance)
                    mean_rewards.append(np.mean(rewards))
                    tot_rew.append(np.sum(rewards))
                    ep_durs.append(ep_dur)
                    mean_com_x_vel = walked_distance / (ep_dur / cfgl.CTRL_FREQ)
                    mean_com_x_vels.append(mean_com_x_vel)
                    break
                else:
                    # we cannot get the reward or walked distance after episode termination,
                    # as when done=True is returned, the env is already resetted.
                    walked_distance = mimic_env.get_walked_distance()
                    # undo reward normalization, don't save last reward
                    reward = reward * np.sqrt(eval_env.ret_rms.var + 1e-8)
                    rewards.append(reward[0])
        if should_record_video:
            imgs = np.array(imgs)
            vid_folder = os.path.join(cfg.save_path, "videos")
            os.makedirs(vid_folder, exist_ok=True)
            fname = os.path.join(vid_folder, "eval_video.gif")
            utils.write_gif_to_disk(imgs, fname, 200)
        # calculate min and mean walked distance
        self.moved_distances = moved_distances
        self.mean_walked_distance = np.mean(moved_distances)
        self.min_walked_distance = np.min(moved_distances)
        self.mean_episode_duration = np.mean(ep_durs)/cfg.ep_dur_max
        self.min_episode_duration = np.min(ep_durs)

        # calculate mean walking speed
        self.mean_walking_speed = np.mean(mean_com_x_vels)
        self.min_walking_speed = np.min(mean_com_x_vels)

        # calculate the average mean reward
        self.mean_reward_means = np.mean(mean_rewards)
        self.tot_reward = np.mean(tot_rew)
        self.logger.record("scalars/tot_reward", self.tot_reward)
        act_rew = [x - y * cfg.alive_bonus for x, y in zip(tot_rew, ep_durs)]
        self.logger.record("scalars/act_reward", np.mean(act_rew))
        self.logger.record("scalars/ep_len", np.mean(ep_durs))

        # normalize it
        self.mean_reward_means = (self.mean_reward_means - cfg.alive_bonus)/cfg.rew_scale

        # determine the amound of stable walks / episodes
        min_required_distance = cfgl.MIN_STABLE_DISTANCE
        runs_below_min_distance = np.where(np.array(moved_distances) < min_required_distance)[0]
        count_runs_reached_min_distance = eval_n_times - len(runs_below_min_distance)
        runs_no_falling = np.where(
            (np.array(ep_durs) == cfg.ep_dur_max) &
            (np.array(moved_distances) >= 0.5*min_required_distance))[0]
        if eval_n_times == cfgl.EVAL_N_TIMES:
            self.failed_eval_runs_indices = runs_below_min_distance.tolist()
        self.count_stable_walks = max(count_runs_reached_min_distance, len(runs_no_falling))
        dt = EVAL_INTERVAL / (
            EVAL_INTERVAL_RARE if self.num_timesteps < EVAL_MORE_FREQUENT_THRES else
            EVAL_INTERVAL_FREQUENT)
        self.summary_score += dt * 4 * self.mean_reward_means ** 2 * (self.count_stable_walks / cfgl.EVAL_N_TIMES) ** 4

        if False: # runs_20m >= 20 and not cfg.is_mod(cfg.MOD_MIRR_QUERY_VF_ONLY):
            cfg.modification += f'/{cfg.MOD_QUERY_VF_ONLY}'
            utils.log('Starting to query VF only!',
                      [f'Stable walks: {count_runs_reached_min_distance}',
                       f'Mean distance: {self.mean_walked_distance}'])

        ## delete evaluation model if stable walking was not achieved yet
        # or too many models were saved already
        were_enough_models_saved = self.n_saved_models >= 5
        # or walking was not human-like
        walks_humanlike = self.mean_reward_means >= 0.5 * (1+self.n_saved_models/10)
        # print('Mean rewards during evaluation of the deterministic model: ', mean_rewards)
        min_dist = int(self.min_walked_distance)
        mean_dist = int(self.mean_walked_distance)
        # walked 10 times at least 20 meters without falling
        has_achieved_stable_walking = min_dist > 20
        # in average stable for 20 meters but not all 20 trials were over 20m
        has_reached_high_mean_distance = mean_dist > 20
        is_stable_humanlike_walking = self.count_stable_walks == eval_n_times and walks_humanlike
        # retain the model if it is good else delete it
        retain_model = is_stable_humanlike_walking and not were_enough_models_saved
        distances_report = [f'Min walked distance: {min_dist}m',
                            f'Mean walked distance: {mean_dist}m']

        if self.best_tot_reward < self.tot_reward:
            utils.save_model(self.model, cfg.save_path, "best", full=False, verbose=True)
            self.best_tot_reward = self.tot_reward

        utils.save_model(self.model, cfg.save_path, "latest", full=False)

        if retain_model:
            utils.log('Saving Model:', distances_report)
            # rename model: add distances to the models names
            dists = f'_min{min_dist}mean{mean_dist}'
            new_model_path = model_path[:-4] + dists +'.zip'
            new_env_path = env_path + dists
            rename(model_path, new_model_path)
            rename(env_path, new_env_path)
            self.n_saved_models += 1
        else:
            utils.log('Deleting Model:', distances_report +
                      [f'Mean step reward: {self.mean_reward_means}',
                       f'Runs below 20m: {runs_below_min_distance}'])
            remove(model_path)
            remove(env_path)

        return is_stable_humanlike_walking


def _save_rews_n_rets(locals):
    # save all rewards and returns of the training, batch wise
    path_rews = cfg.save_path + 'metrics/train_rews.npy'
    path_rets = cfg.save_path + 'metrics/train_rets.npy'

    try:
        # load already saved rews and rets
        rews = np.load(path_rews)
        rets = np.load(path_rets)
        # combine saved with new rews and rets
        rews = np.concatenate((rews, locals['true_reward']))
        rets = np.concatenate((rets, locals['returns']))
    except Exception:
        rews = locals['true_reward']
        rets = locals['returns']

    # save
    np.save(path_rets, np.float16(rets))
    np.save(path_rews, np.float16(rews))




def callback(_locals, _globals):
    """
    Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
    Used to log relevant information during training
    :param _locals: (dict)
    :param _globals: (dict)
    """

    # save all rewards and returns of the training, batch wise
    _save_rews_n_rets(_locals)

    # Log other data about every 200k steps
    # todo: calc as a function of batch for ppo
    #  when updating stable-baselines doesn't provide another option
    #  and check how often TD3 and SAC raise the callback.
    saving_interval = 390 if cfg.use_default_hypers else 6
    n_updates = _locals['update']
    if n_updates % saving_interval == 0:

        model = _locals['self']
        utils.save_pi_weights(model, n_updates)

        # save the model and environment only for every second update (every 400k steps)
        if n_updates % (2*saving_interval) == 0:
            # save model
            model.save(path=cfg.save_path + 'models/model_' + str(n_updates))
            # save env
            env_path = cfg.save_path + 'envs/' + 'env_' + str(n_updates)
            makedirs(env_path)
            # save Running mean of observations and reward
            env = model.get_env()
            env.save_running_average(env_path)
            utils.log("Saved model after {} updates".format(n_updates))

    return True

class SaveVideoCallback(BaseCallback):
    """
    Callback for saving the setpoint tracking plot(the check is done every ``eval_freq`` steps)
    :param eval_env: (gym.Env) The environment used for initialization
    :param n_eval_episodes: (int) The number of episodes to test the agent
    :param eval_freq: (int) Evaluate the agent every eval_freq call of the callback.
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """

    def __init__(self, eval_env, eval_freq=10000, vec_normalise=False, log_dir=None, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.save_path = None
        self.vec_normalise = vec_normalise
        if log_dir is not None:
            self.log_dir = log_dir
            self.save_path = os.path.join(log_dir, "images")

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def preprocess(self, obs):
        if self.vec_normalise:
            return self.model.env.normalize_obs(obs)
        else:
            return obs

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            obs = self.eval_env.reset()
            img = self.eval_env.render("rgb_array")
            imgs = [img]
            done = False
            tot_r = 0.0
            print(f"Begin Evaluation")
            while not done:
                action, _ = self.model.predict(self.preprocess(obs), deterministic=True)
                obs, reward, done, info = self.eval_env.step(action)
                img = self.eval_env.render("rgb_array")
                imgs.append(img)
                tot_r += reward
            print(f"Evaluation Reward: {tot_r}")
            ep_len = len(imgs)
            print(f"Ep Len: {ep_len}")
            imgs = np.array(imgs)

            if self.save_path is not None:
                fname=os.path.join(self.save_path, "eval_video.gif")
                fps = 30 if ep_len < 200 else 60
                utils.write_gif_to_disk(imgs, fname, fps)

        return True
