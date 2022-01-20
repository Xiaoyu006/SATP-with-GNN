import os
import os.path as osp
import time
import torch
from torch_geometric.data import Dataset, DataLoader

# data
class STP_GR_Dataset(Dataset):
    def __init__(self, data_path='/home/xy/stp_data_2021', scenario_names=['stp0750am-0805am']):
        # Initialization
        self.data_path = data_path
        self.scenario_names = scenario_names
        self.all_data_names = os.listdir(self.data_path)


        self.scenario_data_names = [dn for dn in self.all_data_names if dn.split('_')[0] in self.scenario_names]
            
        print('there are {} data pieces for {} on {}'.format(self.__len__(), self.data_path.split('/home/xy/')[1], self.scenario_names))
        super(STP_GR_Dataset).__init__()
    
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.scenario_data_names)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.scenario_data_names[index]
        data_item = torch.load(osp.join(self.data_path, ID))
        # data_item.node_feature = data_item.node_feature.float()
        data_item.x = data_item.node_feature.float()
        data_item.y = data_item.y.float()
        # print(data_item)
        return data_item
    
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    def vis_one_pyg_data(pyg_data):
        pass


    dataset = STP_GR_Dataset(data_path='/home/xy/stp_data_2021', scenario_names=['stp0750am-0805am', 'stp0805am-0820am', 'stp0820am-0835am']) 
    # 'stp0750am-0805am' , 'stp0805am-0820am' 'stp0820am-0835am'
    print(dataset.__getitem__(0).num_edges)
    # print('there are {} data in {} dataset'.format(dataset.__len__(), dataset.scenario_name))
    loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=2)
    # print(dataset.__len__())

    for d in loader:
        tic = time.time()
        print(d)
        tac = time.time()
        # print('used {} sec to load {} data'.format((tac-tic), 128))
        print(max(d.x[:,0,0]))
        print(max(d.x[:,0,1]))
        # print()
        # print(d.edge_index)
        break

    # for i in range(dataset.__len__()):
    #     d = dataset.__getitem__(i)
    #     print(i)
    #     assert d.node_feature.shape[0] >=1
    #     assert d.y.shape[0] >=1
