from .DQN.agent import FloatTensor
import random
from cmath import cos


class Methods:
    ######################################### run episode without reward shaping: epsilon_greedy or greedy #########################################
    def run_episode(ENV, env, agent, step_done, eps, BATCH_SIZE, GAMMA, env_render, log_f, episode, log_f_int_re):
        """Play an epsiode and train

        Args:
            ENV (str) : environment name (Acrobot-v1, MountainCar-v0, LunarLander-v2)
            env (gym.Env): gym environment
            agent (Agent): agent will train and get action
            eps (float): eps-greedy for exploration
            BATCH_SIZE (int): number of training examples utilized in one iteration
            GAMMA (float): discount factor
            env_render (bool): rendering environment at each step or not
            log_f (.csv file): states deep logging
            episode (int): number of current episode
            log_f_int_re (.csv file): rewards deep loging

        Returns:
            int: reward earned in this episode
            int: Time steps in each episode
            int: total of the time steps during the lerning process
        """

        # Reset the environment and get the initial state
        state = env.reset()
        done = False

        #reset Score
        total_reward = 0  # score
        time_step = 0

        loss = 0.0
        icm_loss = 0.0
        int_re = 0.0

        while not done:

            action = agent.get_action(FloatTensor([state]), eps)

            next_state, reward, done, _ = env.step(action.item())

            if ENV == 'LunarLander-v2' or ENV == 'Acrobot-v1':
                next_state_to_print = ['%.2f' % elem for elem in next_state]
                next_state_to_print = ' '.join(next_state_to_print)
            else:
                next_state_to_print = next_state

            time_step += 1
            step_done += 1

            total_reward += reward
            if log_f_int_re != None:
                log_f_int_re.write('{},{},{},{},{:.2f} \n'.format(
                    episode, time_step, step_done, reward, int_re))

            if done:
                reward = -1

                # Store the transition in memory
            agent.replay_memory.push(
                (FloatTensor([state]),
                 action,  # action is already a tensor
                 FloatTensor([reward]),
                 FloatTensor([next_state]),
                 FloatTensor([done])))

            if len(agent.replay_memory) > BATCH_SIZE:

                batch = agent.replay_memory.sample(BATCH_SIZE)

                loss, icm_loss, int_re = agent.learn(batch, GAMMA)

            if log_f != None:
                log_f.write('{},{},{},{},{},{:.2f},{:.2f} \n'.format(
                    episode, time_step, reward, action, next_state_to_print, loss, icm_loss))

            if env_render == True:
                env.render()

            state = next_state

        return total_reward, time_step, step_done

######################################### Mix_in Random reward shaping method for the 3 environments #########################################

    def run_episode_with_MixInRandom_RS(ENV, env, agent, step_done, eps, k, BATCH_SIZE, GAMMA, env_render, log_f, episode, log_f_int_re):
        """Play an epsiode and train

        R' = k*R +(1-k) * U
        U represents a random value from the range of values of the true reward
        k is a uniform value in the range [0.5, 1.0]

        Args:
            ENV (str) : environment name (Acrobot-v1, MountainCar-v0, LunarLander-v2)
            env (gym.Env): gym environment
            agent (Agent): agent will train and get action
            eps (float): eps-greedy for exploration
            k(list): in range [0.5. 1.0], slowly increases
            BATCH_SIZE (int): number of training examples utilized in one iteration
            GAMMA (float): discount factor
            env_render (bool): rendering environment at each step or not
            log_f (.csv file): states deep logging
            episode (int): number of current episode
            log_f_int_re (.csv file): rewards deep loging

        Returns:
            int: reward earned in this episode
            int: Time steps in each episode
            int: shaped reward in this episode
            int: total of the time steps during the lerning process
        """

        # Reset the environment and get the initial state
        state = env.reset()
        done = False

        # Reset the score
        episode_original_reward = 0  # True Score
        episode_shaped_reward = 0

        time_step = 0

        loss = 0.0
        int_re = 0.0

        while not done:

            action = agent.get_action(FloatTensor([state]), eps)

            if ENV == 'MountainCar-v0' or ENV == 'Acrobot-v1':
                U = random.uniform(-1, 1)
            elif ENV == 'LunarLander-v2':
               U = random.randrange(-14, 16)

            next_state, original_reward, done, _ = env.step(action.item())

            if ENV == 'LunarLander-v2' or ENV == 'Acrobot-v1':
                next_state_to_print = ['%.2f' % elem for elem in next_state]
                next_state_to_print = ' '.join(next_state_to_print)
            else:
                next_state_to_print = next_state

            episode_original_reward += original_reward  # originale Score

            shaped_reward = (k*original_reward)+((1-k)*U)
            episode_shaped_reward += shaped_reward  # Shaped Score

            time_step += 1
            step_done += 1
            if log_f_int_re != None:
                log_f_int_re.write('{},{},{},{},{:.2f},{} \n'.format(
                    episode, time_step, step_done, original_reward, int_re, shaped_reward))

            if done:
                original_reward = -1

                # Store the transition in memory
            agent.replay_memory.push(
                (FloatTensor([state]),
                 action,  # action is already a tensor
                 FloatTensor([shaped_reward]),
                 FloatTensor([next_state]),
                 FloatTensor([done])))

            if len(agent.replay_memory) > BATCH_SIZE:

                batch = agent.replay_memory.sample(BATCH_SIZE)

                loss, icm_loss, int_re = agent.learn(batch, GAMMA)

            if log_f != None:
                log_f.write('{},{},{},{},{},{} \n'.format(
                    episode, time_step, original_reward, action, next_state_to_print, loss))

            if env_render == True:
                env.render()

            state = next_state

        return episode_original_reward, time_step, episode_shaped_reward, step_done

    ######################################### the classic Potential based Reward shaping #########################################
    def run_episode_with_PBRS(ENV, env, agent, step_done, eps, BATCH_SIZE, GAMMA, env_render, log_f, episode, log_f_int_re):
       """Play an epsiode and train

           F← gamma * φs' - φs
           Q(s, a) ← Q(s, a) + lr [r + F + gamma maxa Q(s' , a) - Q(s, a)]

       Args:
           ENV (str) : environment name (Acrobot-v1, MountainCar-v0, LunarLander-v2)
           env (gym.Env): gym environment
           agent (Agent): agent will train and get action
           eps (float): eps-greedy for exploration
           BATCH_SIZE (int): number of training examples utilized in one iteration
           GAMMA (float): discount factor
           env_render (bool): rendering environment at each step or not
           log_f (.csv file): states deep logging
           episode (int): number of current episode
           log_f_int_re (.csv file): rewards deep loging

       Returns:
           int: reward earned in this episode
           int: Time steps in each episode
           int: shaped reward in this episode
           int: total of the time steps during the lerning process
       """

       # Reset the environment and get the initial state
       state = env.reset()
       done = False

       # Reset the score
       episode_original_reward = 0
       episode_shaped_reward = 0
       total_reward = 0

       time_step = 0
       po_s = 0

       loss = 0.0
       int_re = 0.0

       while not done:

           action = agent.get_action(FloatTensor([state]), eps)

           next_state, original_reward, done, _ = env.step(action.item())

           if ENV == 'LunarLander-v2' or ENV == 'Acrobot-v1':
               next_state_to_print = ['%.2f' % elem for elem in next_state]
               next_state_to_print = ' '.join(next_state_to_print)
           else:
               next_state_to_print = next_state

           episode_original_reward += original_reward

           po_s = GAMMA * \
               agent.q_target_value(next_state) - agent.q_eval_value(state)

           shaped_reward = original_reward + po_s

           total_reward = shaped_reward
           episode_shaped_reward += total_reward

           time_step += 1
           step_done += 1

           if done:
               original_reward = -1

               # Store the transition in memory
           agent.replay_memory.push(
               (FloatTensor([state]),
                action,  # action is already a tensor
                FloatTensor([total_reward]),
                FloatTensor([next_state]),
                FloatTensor([done])))

           if len(agent.replay_memory) > BATCH_SIZE:

               batch = agent.replay_memory.sample(BATCH_SIZE)

               loss, icm_loss, int_re = agent.learn(batch, GAMMA)

           if log_f != None:
               log_f.write('{},{},{},{},{},{},{} \n'.format(
                   episode, time_step, original_reward, action, next_state_to_print, loss, po_s))
           if log_f_int_re != None:
               log_f_int_re.write('{},{},{},{},{:.2f},{} \n'.format(
                   episode, time_step, step_done, original_reward, int_re, shaped_reward))

           if env_render == True:
               env.render()

           state = next_state

       return episode_original_reward, time_step, episode_shaped_reward, step_done

######################################### the new Potential based Reward shaping #########################################
    def potential_value(original_reward, current_episode_reward, max_episode_reward, min_episode_reward):

        if original_reward != None \
                and max_episode_reward != None \
                and min_episode_reward != None \
                and min_episode_reward != max_episode_reward:
            return ((current_episode_reward - max_episode_reward) / (max_episode_reward - min_episode_reward))
        else:
            return 0

    def run_episode_with_DynamicPBRS(ENV, env, agent, step_done, eps, max_episode_reward, min_episode_reward, BATCH_SIZE, GAMMA, env_render, log_f, episode, log_f_int_re):
       """Play an epsiode and train

       if s is closer to gaol --> F(s,a,s') = r'
       otherwise --> F(s,a,s') = 0

       Args:
           ENV (str) : environment name (Acrobot-v1, MountainCar-v0, LunarLander-v2)
           env (gym.Env): gym environment (CartPole-v0)
           agent (Agent): agent will train and get action
           eps (float): eps-greedy for exploration
           max_episode_reward (float): the maximum of all episode rewards
           min_episode_reward (float): the minimum of all episode rewards
           BATCH_SIZE (int): number of training examples utilized in one iteration
           GAMMA (float): discount factor
           env_render (bool): rendering environment at each step or not
           log_f (.csv file): states deep logging
           episode (int): number of current episode
           log_f_int_re (.csv file): rewards deep loging

       Returns:
           int: reward earned in this episode
           int: Time steps in each episode
           int: shaped reward in this episode
           int: total of the time steps during the lerning process
       """

       # Reset the environment and get the initial state
       state = env.reset()
       done = False
       # Reset the score
       episode_original_reward = 0
       episode_shaped_reward = 0
       reward = 0

       time_step = 0

       loss = 0.0
       int_re = 0.0

       while not done:

           action = agent.get_action(FloatTensor([state]), eps)

           po_1 = Methods.potential_value(
               reward, episode_original_reward, max_episode_reward, min_episode_reward)

           next_state, original_reward, done, _ = env.step(action.item())

           if ENV == 'LunarLander-v2' or ENV == 'Acrobot-v1':
               next_state_to_print = ['%.2f' % elem for elem in next_state]
               next_state_to_print = ' '.join(next_state_to_print)
           else:
               next_state_to_print = next_state

           episode_original_reward += original_reward

           po_2 = Methods.potential_value(
               original_reward, episode_original_reward, max_episode_reward, min_episode_reward)

           po_s = GAMMA * po_2 - po_1

           shaped_reward = original_reward + po_s
           episode_shaped_reward += shaped_reward

           time_step += 1
           step_done += 1

           if done:
               original_reward = -1

               # Store the transition in memory
           agent.replay_memory.push(
               (FloatTensor([state]),
                action,  # action is already a tensor
                FloatTensor([shaped_reward]),
                FloatTensor([next_state]),
                FloatTensor([done])))

           if len(agent.replay_memory) > BATCH_SIZE:

               batch = agent.replay_memory.sample(BATCH_SIZE)

               loss, icm_loss, int_re = agent.learn(batch, GAMMA)

           if log_f != None:
               log_f.write('{},{},{},{},{},{} \n'.format(
                   episode, time_step, reward, action, next_state_to_print, loss))
           if log_f_int_re != None:
               log_f_int_re.write('{},{},{},{},{:.2f},{} \n'.format(
                   episode, time_step, step_done, original_reward, int_re, shaped_reward))

           if env_render == True:
               env.render()

           state = next_state
           reward = original_reward

       return episode_original_reward, time_step, episode_shaped_reward, step_done

######################################### the Goal Bonus reward shaping  method #########################################

    def run_episode_with_GBRS_LL(ENV, env, agent, step_done, eps, N, BATCH_SIZE, GAMMA, env_render, log_f, episode, log_f_int_re):
        """Play an epsiode and train

        if s is closer to gaol --> F(s,a,s') = r'
        otherwise --> F(s,a,s') = 0

        Args:
           ENV (str) : environment name (Acrobot-v1, MountainCar-v0, LunarLander-v2)
           env (gym.Env): gym environment (LunarLander)
           agent (Agent): agent will train and get action
           eps (float): eps-greedy for exploration
           N(list): in ranger [1.0. 0.5], slowly decrease
           BATCH_SIZE (int): number of training examples utilized in one iteration
           GAMMA (float): discount factor
           env_render (bool): rendering environment at each step or not
           log_f (.csv file): states deep logging
           episode (int): number of current episode
           log_f_int_re (.csv file): rewards deep loging

        Returns:
           int: reward earned in this episode
           int: Time steps in each episode
           int: shaped reward in this episode
           int: total of the time steps during the lerning process
        """

        # Reset the environment and get the initial state
        state = env.reset()
        done = False
        # Reset the score
        episode_original_reward = 0
        episode_shaped_reward = 0

        time_step = 0

        loss = 0.0
        int_re = 0.0

        L = [0.0, 0.0]

        while not done:

            distance_to_goal = abs(state[0] - L[0]) + abs(state[1] - L[1])

            action = agent.get_action(FloatTensor([state]), eps)

            next_state, original_reward, done, _ = env.step(action.item())

            if ENV == 'LunarLander-v2' or ENV == 'Acrobot-v1':
                next_state_to_print = ['%.2f' % elem for elem in next_state]
                next_state_to_print = ' '.join(next_state_to_print)
            else:
                next_state_to_print = next_state

            episode_original_reward += original_reward

            new_distance_to_goal = abs(
                next_state[0] - L[0]) + abs(next_state[1] - L[1])

            if new_distance_to_goal < distance_to_goal:

               K = random.uniform(0.2, 0.5)
               F = abs(original_reward * (1-K))

            else:

               F = 0

            shaped_reward = original_reward + F
            episode_shaped_reward += shaped_reward

            time_step += 1
            step_done += 1

            if done:
                original_reward = -1

            # Store the transition in memory
            agent.replay_memory.push(
                (FloatTensor([state]),
                 action,  # action is already a tensor
                 FloatTensor([shaped_reward]),
                 FloatTensor([next_state]),
                 FloatTensor([done])))

            if len(agent.replay_memory) > BATCH_SIZE:

                batch = agent.replay_memory.sample(BATCH_SIZE)
                loss, icm_loss, int_re = agent.learn(batch, GAMMA)

            if log_f != None:
                log_f.write('{},{},{},{},{},{},{} \n'.format(
                    episode, time_step, original_reward, action, next_state_to_print, loss))
            if log_f_int_re != None:
                log_f_int_re.write('{},{},{},{},{:.2f},{} \n'.format(
                    episode, time_step, step_done, original_reward, int_re, shaped_reward))

            if env_render == True:
                env.render()

            state = next_state

        return episode_original_reward, time_step, episode_shaped_reward, step_done

    def run_episode_with_GBRS_MC(ENV, env, agent, step_done, eps, N, BATCH_SIZE, GAMMA, env_render, log_f, episode, log_f_int_re):
        """Play an epsiode and train

        if s is closer to gaol --> F(s,a,s') = r'
        otherwise --> F(s,a,s') = 0

        Args:
           ENV (str) : environment name (Acrobot-v1, MountainCar-v0, LunarLander-v2)
           env (gym.Env): gym environment (MountainCar)
           agent (Agent): agent will train and get action
           eps (float): eps-greedy for exploration
           N(list): in ranger [1.0. 0.5], slowly decrease
           BATCH_SIZE (int): number of training examples utilized in one iteration
           GAMMA (float): discount factor
           env_render (bool): rendering environment at each step or not
           log_f (.csv file): states deep logging
           episode (int): number of current episode
           log_f_int_re (.csv file): rewards deep loging

        Returns:
           int: reward earned in this episode
           int: Time steps in each episode
           int: shaped reward in this episode
           int: total of the time steps during the lerning process
        """

        # Reset the environment and get the initial state
        state = env.reset()
        done = False
        # Reset the score
        episode_original_reward = 0
        episode_shaped_reward = 0

        time_step = 0

        loss = 0.0
        int_re = 0.0

        goal = [0.5]

        while not done:

            distance_from_goal = abs(goal[0]-state[0])

            action = agent.get_action(FloatTensor([state]), eps)

            next_state, original_reward, done, _ = env.step(action.item())

            if ENV == 'LunarLander-v2' or ENV == 'Acrobot-v1':
                next_state_to_print = ['%.2f' % elem for elem in next_state]
                next_state_to_print = ' '.join(next_state_to_print)
            else:
                next_state_to_print = next_state

            episode_original_reward += original_reward

            next_distance_from_goal = abs(goal[0]-next_state[0])

            if next_distance_from_goal < distance_from_goal:

                F = abs(original_reward * (1-N))

            else:

                F = 0

            shaped_reward = original_reward + F
            episode_shaped_reward += shaped_reward

            time_step += 1
            step_done += 1

            if done:
                original_reward = -1

            # Store the transition in memory
            agent.replay_memory.push(
                (FloatTensor([state]),
                 action,  # action is already a tensor
                 FloatTensor([shaped_reward]),
                 FloatTensor([next_state]),
                 FloatTensor([done])))

            if len(agent.replay_memory) > BATCH_SIZE:

                batch = agent.replay_memory.sample(BATCH_SIZE)
                loss, icm_loss, int_re = agent.learn(batch, GAMMA)

            if log_f != None:
                log_f.write('{},{},{},{},{},{} \n'.format(
                    episode, time_step, original_reward, action, next_state_to_print, loss))
            if log_f_int_re != None:
                log_f_int_re.write('{},{},{},{},{:.2f},{} \n'.format(
                    episode, time_step, step_done, original_reward, int_re, shaped_reward))

            if env_render == True:
                env.render()

            state = next_state

        return episode_original_reward, time_step, episode_shaped_reward, step_done

    def run_episode_with_GBRS_AC(ENV, env, agent, step_done, eps, N, BATCH_SIZE, GAMMA, env_render, log_f, episode, log_f_int_re):
        """Play an epsiode and train

        if s is closer to gaol --> F(s,a,s') = r'
        otherwise --> F(s,a,s') = 0

        Args:
           ENV (str) : environment name (Acrobot-v1, MountainCar-v0, LunarLander-v2)
           env (gym.Env): gym environment (Acrobot)
           agent (Agent): agent will train and get action
           eps (float): eps-greedy for exploration
           N(list): in ranger [1.0. 0.5], slowly decrease
           BATCH_SIZE (int): number of training examples utilized in one iteration
           GAMMA (float): discount factor
           env_render (bool): rendering environment at each step or not
           log_f (.csv file): states deep logging
           episode (int): number of current episode
           log_f_int_re (.csv file): rewards deep loging

        Returns:
           int: reward earned in this episode
           int: Time steps in each episode
           int: shaped reward in this episode
           int: total of the time steps during the lerning process
        """

        # Reset the environment and get the initial state
        state = env.reset()
        done = False
        # Reset the score
        episode_original_reward = 0
        episode_shaped_reward = 0

        time_step = 0

        loss = 0.0
        int_re = 0.0

        highest_position = (-cos(state[0]) - cos(state[1] + state[0]))

        while not done:

            action = agent.get_action(FloatTensor([state]), eps)

            next_state, original_reward, done, _ = env.step(action.item())

            if ENV == 'LunarLander-v2' or ENV == 'Acrobot-v1':
                next_state_to_print = ['%.2f' % elem for elem in next_state]
                next_state_to_print = ' '.join(next_state_to_print)
            else:
                next_state_to_print = next_state

            episode_original_reward += original_reward

            next_diff_from_goal = (-cos(next_state[0]) -
                                   cos(next_state[1] + next_state[0]))

            if next_diff_from_goal.real > highest_position.real:
                highest_position = next_diff_from_goal

                F = abs(original_reward * (1-N))

            else:

                F = 0

            shaped_reward = original_reward + F
            episode_shaped_reward += shaped_reward

            time_step += 1
            step_done += 1

            if done:
                original_reward = -1

            # Store the transition in memory
            agent.replay_memory.push(
                (FloatTensor([state]),
                 action,  # action is already a tensor
                 FloatTensor([shaped_reward]),
                 FloatTensor([next_state]),
                 FloatTensor([done])))

            if len(agent.replay_memory) > BATCH_SIZE:

                batch = agent.replay_memory.sample(BATCH_SIZE)
                loss, icm_loss, int_re = agent.learn(batch, GAMMA)

            if log_f != None:
                log_f.write('{},{},{},{},{},{} \n'.format(
                    episode, time_step, original_reward, action, next_state_to_print, loss))
            if log_f_int_re != None:
                log_f_int_re.write('{},{},{},{},{:.2f},{} \n'.format(
                    episode, time_step, step_done, original_reward, int_re, shaped_reward))

            if env_render == True:
                env.render()

            state = next_state

        return episode_original_reward, time_step, episode_shaped_reward, step_done
