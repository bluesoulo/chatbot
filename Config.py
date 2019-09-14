import torch
class Config:
    '''
    Chatbot模型参数
    '''
    model_name = 'chatbot'
    save_dir = './checkpoint/'
    max_input_length = 150 #输入的最大句子长度
    max_generate_length = 60 #生成的最大句子长度
    prefix = 'checkpoints/chatbot'  #模型断点路径前缀
    model_ckpt  = '/home/zhangjinjie/torch_chatbot/checkpoint/model/chatbot/3-3_256/5000_backup_bidir_model.tar'
    pretrained_embedding_path = '/home/zhangjinjie/knowledge-driven-dialogue-master2/knowledge-driven-dialogue-master/generative_pt/sgns.baidubaike.bigram-char'
    '''
    训练超参数
    '''
    embedding_dim =300
    batch_size = 64
    #bidirectional = True #Encoder-RNN是否双向
    hidden_size = 256
    embedding_dim = 300
    method = 'dot' #attention method
    dropout = 0.1 #是否使用dropout
    clip = 50.0 #梯度裁剪阈值
    num_layers = 3 #Encoder-RNN层数
    learning_rate = 1e-3
    teacher_forcing_ratio = 1 #teacher_forcing比例
    decoder_learning_ratio = 5.0
    '''
    训练周期信息
    '''
    epoch = 5000
    print_every = 100 #每隔print_every个Iteration打印一次
    save_every = 1000 #每隔save_every个Epoch打印一次

