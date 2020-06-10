import torch
from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from torch.utils.data import DataLoader,Dataset
from argoverse.map_representation.map_api import ArgoverseMap
import dgl
import numpy as np

def get_lane_centerlines(argoverse_data,avm):
    '''
    根据车辆位置信息和地图，获取周围的车道信息
    '''
    df = argoverse_data.seq_df
    agent_obs_traj = argoverse_data.agent_traj[:50]
    time_list = np.sort(np.unique(df["TIMESTAMP"].values))
    city_name = df["CITY_NAME"].values[0]
    x_min = min(agent_obs_traj[:,0])
    x_max = max(agent_obs_traj[:,0])
    y_min = min(agent_obs_traj[:,1])
    y_max = max(agent_obs_traj[:,1])
    seq_lane_props = avm.city_lane_centerlines_dict[city_name]
    lane_centerlines = []
    for lane_id, lane_props in seq_lane_props.items():
        lane_cl = lane_props.centerline
        if (
                np.min(lane_cl[:, 0]) < x_max
                and np.min(lane_cl[:, 1]) < y_max
                and np.max(lane_cl[:, 0]) > x_min
                and np.max(lane_cl[:, 1]) > y_min
        ):
            lane_centerlines.append(lane_cl)
    return lane_centerlines

def compose_graph(lane,label):
    '''
    输入的是车道向量
    把车道组织成图,并返回结点特征((x1,y1),(x2,y2),label)
    '''
    nodeN = lane.shape[0]-1
    features = torch.zeros(nodeN,5)
    graph = dgl.DGLGraph()
    graph.add_nodes(nodeN)
    for i in range(nodeN):
        features[i][0] = lane[i][0]
        features[i][1] = lane[i][1]
        features[i][2] = lane[i+1][0]
        features[i][3] = lane[i+1][1]
        features[i][4] = label#torch.tensor(list(lane[i])+list(lane[i+1])+[label])
    src = []
    dst = []
    for i in range(nodeN):
        for j in range(nodeN):
            if i != j:
                src.append(i)
                dst.append(j)
    graph.add_edges(src,dst)
    graph.ndata['v_feature'] = features
    return graph,features

def collate(samples):
    datas,labels = map(list,zip(*samples))
    AgentGraph = [data['Agent'] for data in datas]
    batched_graph = dgl.batch(AgentGraph)
    map_set = []
    feature_set = []
    for data in datas:
        for i in range(len(data['Map'])):
            map_set.append(data['Map'][i])
            feature_set.append(data['Mapfeature'][i])
    new_data = {}
    new_data['Map'] = map_set
    new_data['Mapfeature'] = feature_set
    new_data['Agent'] = batched_graph
    new_data['Agentfeature'] = batched_graph.ndata['v_feature']
    #new_label = []
    #for l in labels:
        #new_label += list(l.flatten())
    return new_data,torch.tensor(labels).reshape(-1,60)

class VectorNetDataset(Dataset):
    '''
    VectorNet数据集
    '''
    def __init__(self,root,train = True,test = False):
        '''
        根据路径获得数据，并根据训练、验证、测试划分数据
        train_data 和 test_data路径分开
        '''
        self.test = test
        afl = ArgoverseForecastingLoader(root)
        self.avm = ArgoverseMap()
        al = []
        for d in afl:
            al.append(d)
        n = len(al)
        if self.test == True:
            self.afl = al
        elif train:
            self.afl = al[:int(0.7*n)]
        else:
            self.afl = al[int(0.7*n):]
    
    def __getitem__(self,index):
        '''
        从csv创建图输入模型
        '''
        data = {}
        argoverse_forecasting_data = self.afl[index]
        lane_centerlines = get_lane_centerlines(argoverse_forecasting_data,self.avm)
        agent_obs_traj = argoverse_forecasting_data.agent_traj
        data['centerAgent'] = agent_obs_traj[19]
        #把车道组织成向量和图
        map_set = []
        map_feature = []
        for lane in lane_centerlines:
            lane = (lane - agent_obs_traj[19])#/np.array([x_max-x_min,y_max-y_min])
            graph,features = compose_graph(lane,len(map_set))
            map_set.append(graph)
            map_feature.append(features)
        agent_obs_traj_norm = (agent_obs_traj - agent_obs_traj[19])#/np.array([x_max-x_min,y_max-y_min])
        graph,features = compose_graph(agent_obs_traj_norm[:20],len(map_set))
        data['Map'] = map_set
        data['Mapfeature'] = map_feature
        data['Agent'] = graph
        data['Agentfeature'] = features
        label = agent_obs_traj_norm[20:50]
        return data,label.flatten()
    
    def __len__(self):
        return len(self.afl)
