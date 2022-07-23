
class network(nn.Module):
    
    def __init__(self , num_state , num_action):
        
        super().__init__()
        self.fc1 = nn.Linear(num_state , 1024 )
        self.fc2 = nn.Linear(1024 , 512)
        self.fc3 = nn.Linear(512 , 256)
        self.fc4 = nn.Linear(256 , 128)
        self.fc5 = nn.Linear(128 , 64)
        self.fc6 = nn.Linear(64 , 32)
        self.fc7 = nn.Linear(32 , 16)
        self.out = nn.Linear(16 , num_action )
    
    def forward(self , x):
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = self.out(x)
        return x

class ReplayBuffer(object):
    '''
        
        This code is copied from openAI baselines
        https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
        '''
    def __init__(self, size):
        self._storage = []
        self._maxsize = size
        self._next_idx = 0
    
    def __len__(self):
        return len(self._storage)
    
    def add(self, obs_t, action, reward, obs_tp1, done):
        
        data = (obs_t, action, reward, obs_tp1, done)
        
        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize
    
    def _encode_sample(self, idxes , dtype = np.float32):
        
        
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False,dtype=dtype))
            actions.append(np.array(action, copy=False,dtype=np.long))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False,dtype=dtype))
            dones.append(done)
        return np.array(obses_t,dtype=dtype), np.array(actions , dtype = np.long), \
    np.array(rewards  ,dtype=dtype), np.array(obses_tp1,dtype=dtype), np.array(dones , dtype = bool)
    
    
    def sample(self, batch_size):
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)


class Agent():
    
    def __init__(self , num_state , num_action):
        
        
        
        self.policy_network = network(num_state , num_action)
        self.target_network = network(num_state , num_action)
        
        self.target_network.load_state_dict(self.policy_network.state_dict())
        
        self.steps_done = 0
        self.num_state = num_state
        self.num_action = num_action
        
        self.EPS_END = 0.05
        self.EPS_START = 0.999
        
        self.EPS_DECAY = 30000
        self.batch_size = 64
        self.buffer = ReplayBuffer( 4000 )
        #         self.optimizer = torch.optim.Adam(self.policy_network.parameters()   , amsgrad=True)
        self.optimizer = torch.optim.AdamW(self.policy_network.parameters(),lr=0.00002)
    def take_action(self , x , adj_array, is_testing = False ) :
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * self.steps_done / self.EPS_DECAY)
        if(self.steps_done%3000) == 0:
            print(eps_threshold)
        x = x.astype(np.float32)
        x = torch.from_numpy(x)
        rand_val = np.random.uniform()
        if rand_val > eps_threshold or is_testing == True:
            val = self.policy_network(x)
            mask_list = [0] * self.num_state
            for item in adj_array :
                mask_list[item] = 1
            for i in range(len(mask_list)):
                if(mask_list[i] == 0):
                    val[i] = float('-Infinity')
            action = torch.argmax(val).item()
        
        
        else:
            action = np.random.choice(adj_array)
        
        if is_testing == False:
            self.steps_done += 1
        
        return action
    
    
    def store_transition(self, state , action , reward , next_state , done ):
        
        self.buffer.add(state , action , reward , next_state , done)
    
    def update_parameters(self):
        
        if len(self.buffer) < self.batch_size:
            return
        
        loss_fn = torch.nn.MSELoss(reduction = 'mean')
        
        batch = self.buffer.sample(self.batch_size)
        states , actions , rewards , next_states , dones = batch
        states = torch.from_numpy(states)
        actions = torch.from_numpy(actions).view(-1,1)
        rewards = torch.from_numpy(rewards)
        next_states = torch.from_numpy(next_states)
        actions = actions.long()
        
        non_final_mask = torch.tensor(tuple(map(lambda s : s != True, dones)),dtype = torch.bool)
        non_final_next_state = next_states[non_final_mask]
        
        pred_q = self.policy_network(states).gather(1 , actions).view(-1)
        next_state_value = torch.zeros(self.batch_size).detach()
        
        D_action = self.policy_network(non_final_next_state).argmax(1).view(-1,1)
        next_state_value[non_final_mask] = self.target_network(non_final_next_state).gather(1 , D_action).view(-1)
        expected_q = (next_state_value + rewards).detach()
        
        loss = loss_fn(pred_q , expected_q)
        self.optimizer.zero_grad()
        loss.backward()
                self.optimizer.step()
            def update_target_weight(self):
                self.target_network.load_state_dict(self.policy_network.state_dict())




file = open('data_20.txt')
lines = file.readlines()
num_nodes = int(lines[0])
num_edges = int(lines[1])
agent = Agent(num_nodes , num_nodes)
reward_history = []
cost = 3000
for e in tqdm(range(150)):
    #generator()
    #     file = open('data.txt')
    #     lines = file.readlines()
    #     num_nodes = int(lines[0])
    #     num_edges = int(lines[1])
    print(cost)
    if(cost <= 10000) and e > 30:
        break
    nodes = []
    state = []
    edge_len = [[0]*num_nodes for i in range(num_nodes)]
    god_map = [[0]*num_nodes for i in range(num_nodes)]
    nodes_dis = []
    nodes_dis2 = [[0]*num_nodes for i in range(num_nodes)]
    on_nodes = []
    info_speed = []
    features = []
    prev_features = []
    Adj = [[0]*num_nodes for i in range(num_nodes)]
    
    for i in range(num_nodes):
        state.append([])
        nodes_dis.append([])
        on_nodes.append([])
        curLine = lines[i+2].strip().split(" ")
        intLine = list(map(int, curLine))
        nodes.append([intLine[0], intLine[1], intLine[2]])
        for j in range(i-1, -1, -1):
            dis = distance(nodes, i, j)
            nodes_dis[i].append((j, dis))
            nodes_dis[j].append((i, dis))
            nodes_dis2[i][j] = dis
            nodes_dis2[j][i] = dis
    for i in range(len(nodes_dis)):
        nodes_dis[i].sort(key=lambda nodes_dis: nodes_dis[1])
    
    for i in range(num_edges):
        curLine = lines[i+2+num_nodes].strip().split(" ")
        intLine = list(map(int, curLine))
        state[intLine[0]].append(intLine[1])
        state[intLine[1]].append(intLine[0])
        edge_len[intLine[0]][intLine[1]] = intLine[2]
        edge_len[intLine[1]][intLine[0]] = intLine[2]
        Adj[intLine[0]][intLine[1]] = 1
        Adj[intLine[1]][intLine[0]] = 1
    Adj = torch.FloatTensor(Adj)
    k_agents = int(lines[2+num_nodes+num_edges])
    flag = [[0]*k_agents for i in range(k_agents)]
    for k in range(k_agents):
        features.append([])
        prev_features.append([])
        for i in range(num_nodes):
            features[k].append([])
            prev_features[k].append([])
            for j in range(num_nodes):
                features[k][i].append(0)
                prev_features[k][i].append(0)

now_point = [0] * k_agents
speed = [0]*k_agents
target = [0]*k_agents
location = [0]*k_agents
#     x_agent = []
#     y_agent = []

for i in range(k_agents):
    #         x_agent.append([])
    #         y_agent.append([])
    info_speed.append([])
    curLine = lines[i+3+num_nodes+num_edges].strip().split(" ")
    intLine = list(map(int, curLine))
    #now_point[intLine[0]] = intLine[1]
    now_point[i] = random.randint(0, num_nodes-1)
        speed[intLine[0]] = intLine[2]

    t_constraint = int(lines[3+num_nodes+num_edges+k_agents])

#     print(now_point)
#     print(state)
#     print(speed)
#     print(edge_len)

history_route = []
state_map = []
for i in range(k_agents):
    history_route.append([])
    history_route[i].append(now_point[i])
    on_nodes[now_point[i]].append(i)
    state_map.append([])
    for i in range(k_agents):
        state_map[i] = [[0] * num_nodes for i in range(num_nodes)]
    
    finish_count = 0
    cost = 0
    pre_step = [0]*k_agents
    info_clustering = clustering()
    reward_sum = 0.0
    while finish_count < num_edges:
        communication()
        #         for i in range(k_agents):
        #             for state_index in range(num_nodes):
        #                 for state_index2 in range(num_nodes):
        #                     if(state_map[i][state_index][state_index2] >= 1):
        #                         features[i][state_index][state_index2] = 1
        cost+=1
        for i in range(k_agents):
            list_x = []
            list_y = []
            if target[i]==0:
                
                for state_index in range(num_nodes):
                    for state_index2 in range(num_nodes):
                        if(state_map[i][state_index][state_index2] >= 1):
                            features[i][state_index][state_index2] = 1
                reward = 0
                pre_step[i] = now_point[i]
                action = agent.take_action(np.array(features[i][now_point[i]]),state[pre_step[i]])
                for index_i in range(len(features[i])):
                    for index_j in range(len(features[i][index_i])):
                        prev_features[i][index_i][index_j] = features[i][index_i][index_j]
                now_point[i] = action
                if(state_map[i][now_point[i]][pre_step[i]] == 0):
                    reward += 1
                #                 else:
                #                     reward -= features[i][now_point[i]][pre_step[i]] * 0.001
                reward -= edge_len[pre_step[i]][now_point[i]] * 0.001
                if(finish_count >= num_edges - 1 and god_map[now_point[i]][pre_step[i]]==0):
                    done = True
                    reward += 8
                    if(cost <= 9100):
                        reward += 25
                else:
                    done = False
                reward_sum += reward
                
                if god_map[now_point[i]][pre_step[i]]==0:
                    finish_count+=1
                god_map[now_point[i]][pre_step[i]] += 1
                god_map[pre_step[i]][now_point[i]] += 1
                state_map[i][now_point[i]][pre_step[i]] += 1
                state_map[i][pre_step[i]][now_point[i]] += 1
                
                features[i][pre_step[i]][now_point[i]] = 1
                features[i][now_point[i]][pre_step[i]] = 1
                agent.store_transition(np.array(prev_features[i][pre_step[i]]) , action , reward , np.array(features[i][pre_step[i]]), done)
                agent.update_parameters()
                
                target[i] = edge_len[now_point[i]][pre_step[i]]/speed[i]
            location[i]+=1
            draw_flag = 0
            while location[i]>=target[i]:
                #                 state_map[i][now_point[i]][pre_step[i]] += 1
                #                 state_map[i][pre_step[i]][now_point[i]] += 1
                history_route[i].append(now_point[i])
                #                 if god_map[now_point[i]][pre_step[i]]==0:
                #                     finish_count+=1
                #                 god_map[now_point[i]][pre_step[i]] += 1
                #                 god_map[pre_step[i]][now_point[i]] += 1
                
                #                 if(draw_flag == 0):
                #                     pre_x = ((location[i]-1)/target[i])*nodes[now_point[i]][1] + ((target[i] - (location[i]-1))/target[i])*nodes[pre_step[i]][1]
                #                     pre_y = ((location[i]-1)/target[i])*nodes[now_point[i]][2] + ((target[i] - (location[i]-1))/target[i])*nodes[pre_step[i]][2]
                #                     now_x = nodes[now_point[i]][1]
                #                     now_y = nodes[now_point[i]][2]
                #                     list_x.append([pre_x, now_x])
                #                     list_y.append([pre_y, now_y])
                #                 else:
                #                     pre_x = nodes[pre_step[i]][1]
                #                     pre_y = nodes[pre_step[i]][2]
                #                     now_x = nodes[now_point[i]][1]
                #                     now_y = nodes[now_point[i]][2]
                #                     list_x.append([pre_x, now_x])
                #                     list_y.append([pre_y, now_y])
                
                for state_index in range(num_nodes):
                    for state_index2 in range(num_nodes):
                        if(state_map[i][state_index][state_index2] >= 1):
                            features[i][state_index][state_index2] = 1
                reward = 0
                pre_step[i] = now_point[i]
                action = agent.take_action(np.array(features[i][now_point[i]]),state[pre_step[i]])
                for index_i in range(len(features[i])):
                    for index_j in range(len(features[i][index_i])):
                        prev_features[i][index_i][index_j] = features[i][index_i][index_j]
                now_point[i] = action
                if(state_map[i][now_point[i]][pre_step[i]] == 0):
                    reward += 1
                #                 else:
                #                     reward -= features[i][now_point[i]][pre_step[i]] * 0.001
                reward -= edge_len[pre_step[i]][now_point[i]] * 0.001
                if(finish_count >= num_edges - 1 and god_map[now_point[i]][pre_step[i]]==0):
                    done = True
                    reward += 8
                    if(cost <= 9100):
                        reward += 25
                else:
                    done = False
                reward_sum += reward

                if god_map[now_point[i]][pre_step[i]]==0:
                    finish_count+=1
                god_map[now_point[i]][pre_step[i]] += 1
                god_map[pre_step[i]][now_point[i]] += 1
                state_map[i][now_point[i]][pre_step[i]] += 1
                state_map[i][pre_step[i]][now_point[i]] += 1

                features[i][pre_step[i]][now_point[i]] = 1
                features[i][now_point[i]][pre_step[i]] = 1
                agent.store_transition(np.array(prev_features[i][pre_step[i]]) , action , reward , np.array(features[i][pre_step[i]]), done)
                agent.update_parameters()

                location[i] = location[i]-target[i]
                target[i] = edge_len[now_point[i]][pre_step[i]]/speed[i]
                if i in on_nodes[pre_step[i]]:
                    on_nodes[pre_step[i]].remove(i)
                draw_flag = 1
            #             if (draw_flag == 1):
            #                 pre_x = nodes[pre_step[i]][1]
            #                 pre_y = nodes[pre_step[i]][2]
            #                 now_x = ((location[i])/target[i])*nodes[now_point[i]][1] + ((target[i] - location[i])/target[i])*nodes[pre_step[i]][1]
            #                 now_y = ((location[i])/target[i])*nodes[now_point[i]][2] + ((target[i] - location[i])/target[i])*nodes[pre_step[i]][2]
            #                 list_x.append([pre_x, now_x])
            #                 list_y.append([pre_y, now_y])
            #                 x_agent[i].append(list_x)
            #                 y_agent[i].append(list_y)
            #             else:
            #                 pre_x = ((location[i]-1)/target[i])*nodes[now_point[i]][1] + ((target[i] - (location[i]-1))/target[i])*nodes[pre_step[i]][1]
            #                 pre_y = ((location[i]-1)/target[i])*nodes[now_point[i]][2] + ((target[i] - (location[i]-1))/target[i])*nodes[pre_step[i]][2]
            #                 now_x = ((location[i])/target[i])*nodes[now_point[i]][1] + ((target[i] - location[i])/target[i])*nodes[pre_step[i]][1]
            #                 now_y = ((location[i])/target[i])*nodes[now_point[i]][2] + ((target[i] - location[i])/target[i])*nodes[pre_step[i]][2]
            #                 x_agent[i].append([pre_x, now_x])
            #                 y_agent[i].append([pre_y, now_y])

            if (location[i]/target[i]) > 0.5:
                if i not in on_nodes[now_point[i]]:
                    on_nodes[now_point[i]].append(i)
            else:
                if i not in on_nodes[pre_step[i]]:
                    on_nodes[pre_step[i]].append(i)
    reward_history.append(reward_sum)
    if e  % 1 == 0:
        print(reward_sum)
    if e > 0 and e % 20 == 0:
        agent.update_target_weight()

