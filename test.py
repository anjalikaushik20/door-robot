import os
import robosuite as suite
from robosuite.wrappers import GymWrapper
from td3_torch import Agent
import time

if __name__ == "__main__":
    
    if not os.path.exists("tmp/td3"):
        os.makedirs("tmp/td3")
        
    env_name = "Door"
    
    env = suite.make(
        env_name=env_name,
        robots=["Panda"],
        controller_configs = suite.load_controller_config(default_controller="JOINT_VELOCITY"),
        has_renderer=True,
        use_camera_obs=False,
        render_camera="frontview",
        has_offscreen_renderer=True,
        horizon=300,
        reward_shaping=True,
        control_freq=20,
    )
    
    env = GymWrapper(env)
    
    actor_learning_rate = 0.001
    critic_learning_rate = 0.001
    batch_size = 128
    layer1_size = 256
    layer2_size = 128
    
    agent = Agent(
        actor_learning_rate=actor_learning_rate,
        critic_learning_rate=critic_learning_rate,
        batch_size=batch_size,
        layer1_size=layer1_size,
        layer2_size=layer2_size,
        env=env,
        tau=0.005,
        input_dims=env.observation_space.shape,
        n_actions=env.action_space.shape[0],
    )
    
    n_games = 3
    best_score = 0
    episode_identifier = f"0 - actor_lr: {actor_learning_rate}, critic_lr: {critic_learning_rate}, batch_size: {batch_size}, layer1_size: {layer1_size}, layer2_size: {layer2_size}"
    agent.load_models()
    
    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.choose_action(observation, validation=True)
            next_observation, reward, done, info = env.step(action)
            env.render()
            score += reward
            observation = next_observation
            time.sleep(0.03)

            
        print(f"Episode {i}, Score: {score}")