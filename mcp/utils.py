import os

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from tqdm.auto import tqdm


def evaluate(env, model):
    done = False
    obs = env.reset()
    pbar = tqdm(total=1000)
    rew = []
    while not np.any(done):
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        rew.append(rewards)
        pbar.update(1)
        done = dones
    rew = np.array(rew)
    pbar.close()
    print(rew.sum(0))


def encode_gif(frames, fps):
    from subprocess import PIPE, Popen

    h, w, c = frames[0].shape
    pxfmt = {1: "gray", 3: "rgb24"}[c]
    cmd = " ".join(
        [
            "ffmpeg -y -f rawvideo -vcodec rawvideo",
            f"-r {fps:.02f} -s {w}x{h} -pix_fmt {pxfmt} -i - -filter_complex",
            "[0:v]split[x][z];[z]palettegen[y];[x]fifo[x];[x][y]paletteuse",
            f"-r {fps:.02f} -f gif -",
        ]
    )
    proc = Popen(cmd.split(" "), stdin=PIPE, stdout=PIPE, stderr=PIPE)
    for image in frames:
        proc.stdin.write(image.tobytes())
    out, err = proc.communicate()
    if proc.returncode:
        raise IOError("\n".join([" ".join(cmd), err.decode("utf8")]))
    del proc
    return out


def write_gif_to_disk(frames, filename, fps=10):
    """
    frame: np.array of shape TxHxWxC
    """
    try:
        frames = encode_gif(frames, fps)
        with open(filename, "wb") as f:
            f.write(frames)
        tqdm.write(f"GIF saved to {filename}")
    except Exception as e:
        tqdm.write(frames.shape)
        tqdm.write("GIF Saving failed.", e)


class SaveVideoCallback(BaseCallback):
    """
    Callback for evaluating the model and (optionally) saving a gif of the performance (the check is done every ``eval_freq`` steps)
    :param eval_env: (gym.Env) The environment used for initialization
    :param eval_freq: (int) Evaluate the agent every eval_freq call of the callback.
    :param video_freq: (int) Save gif of performance every video_freq call of the callback.
    :param vec_normalise: (bool) Whether the env should use a VecNormalise instance.
    :param log_dir: (str) Path to the folder where the gifs and images will be saved.
    :param verbose: (int)
    """

    def __init__(
        self,
        eval_env,
        eval_freq=10000,
        video_freq=10000,
        vec_normalise=False,
        log_dir=None,
        verbose=1,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        assert isinstance(eval_freq, int)
        assert isinstance(video_freq, int)
        assert eval_freq <= video_freq and video_freq % eval_freq == 0
        self.eval_freq = eval_freq
        self.video_freq = video_freq
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
            if self.n_calls % self.video_freq == 0:
                img = self.eval_env.render("rgb_array")
                imgs = [img]
            done = False
            tot_r = 0.0
            weights = []
            print(f"Begin Evaluation")
            pred_weights = False
            if hasattr(self.model.policy, "predict_weights"):
                pred_weights = True
            pbar = tqdm(total=1000)
            i = 0
            while not done:
                action, _ = self.model.predict(self.preprocess(obs), deterministic=True)
                if pred_weights:
                    weight = self.model.policy.predict_weights(obs)
                    weights.append(weight)
                obs, reward, done, info = self.eval_env.step(action)
                if self.n_calls % self.video_freq == 0:
                    img = self.eval_env.render("rgb_array")
                    imgs.append(img)
                tot_r += reward
                pbar.update(1)
                i += 1
            pbar.close()

            print(f"Evaluation Reward: {tot_r}")

            ep_len = i
            print(f"Ep Len: {ep_len}")

            if pred_weights:
                weights = np.array(weights).squeeze(1)
                fname = os.path.join(self.save_path, "weights.npy")
                np.save(fname, weights)
                for i in range(weights.shape[1]):
                    plt.plot(weights[:, i], label=f"Model {i}")
                plt.xlim(0, ep_len)
                plt.ylim(0, 1)
                plt.title("Weights assigned to primitives")
                plt.tight_layout()
                plt.legend()

                fname = os.path.join(self.save_path, "weights.jpg")
                plt.savefig(fname, bbox_inches="tight", dpi=120)
                plt.close()

            if self.save_path is not None and self.n_calls % self.video_freq == 0:
                imgs = np.array(imgs)
                fname = os.path.join(self.save_path, "eval_video.gif")
                fps = 30 if ep_len < 200 else 60
                write_gif_to_disk(imgs, fname, fps)

        return True
