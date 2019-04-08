import numpy as np 

pooled_height = 2                                       # 每个区域池化之后的高度
input_data = np.random.randint(1, 255,(1,64,50,50))       # image 
rois = np.asarray([[0,10,10,20,20], [1,20,20,40,40],[2,1,2,30,7]])

batch, num_channels, height, weight = input_data.shape

num_rois = rois.shape[0]

max_ratio = max((rois[:,3] - rois[:,1]) / (rois[:,4] - rois[:,2]))      # 求出最大的宽度
max_pooled_weight = int(np.ceil(max_ratio) * pooled_height)
output = np.zeros((num_rois, num_channels, pooled_height, max_pooled_weight))       # 300×64×2×10;10是序列的长度
argmax = np.ones((num_rois, num_channels, pooled_height, max_pooled_weight)) * -1       # 最大值的索引

for n in range(num_rois):
    roi_start_w = np.round(rois[n, 1])
    roi_start_h = np.round(rois[n, 2])

    rois_weight = np.max([rois[n,3] - rois[n,1], 1])                     # 每个区域的宽度
    rois_height = np.max([rois[n,4] - rois[n,2], 1])                     # 每个区域的高度

    pooled_weight = np.ceil(np.float(rois_weight) / rois_height * pooled_height)        # 和rois等比例的池化
    pooled_weight = int(pooled_weight)

    bin_size_w = np.float(rois_weight) / pooled_weight                  # 每个bin块的宽度
    bin_size_h = np.float(rois_height) / pooled_height                  # 每个bin块的高度

    for c in range(num_channels):
        for ph in range(pooled_height):                                 # numpy的矩阵展开的时候是行优先的
            for pw in range(pooled_weight):
                
                hstart = np.floor(ph * bin_size_h)
                wstart = np.floor(pw * bin_size_w)
                hend = np.ceil((ph + 1) * bin_size_h)
                wend = np.ceil((pw + 1) * bin_size_w)

                hstart = min(max(hstart + roi_start_h, 0),height)           # 将每个bin限制在图片的尺寸范围内
                hend = min(max(hend + roi_start_h, 0), height)
                wstart = min(max(wstart + roi_start_w, 0), weight)
                wend = min(max(wend + roi_start_w, 0), weight)
                
                hstart, hend, wstart, wend = map(lambda x: int(x), [hstart, hend, wstart, wend])

                output[n, c, ph, pw] = np.max(input_data[:, c, hstart:hend, wstart:wend])    # 最大值
                argmax[n, c, ph, pw] = np.argmax(input_data[:,c, hstart:hend, wstart:wend])  # 最大值的索引//


# backward反向传播
grad_input = np.random.randn(num_rois, num_channels, pooled_height, max_pooled_weight)       # 梯度
mask = argmax.copy()                
mask[0 <= mask] = 1
mask[mask < 0] = 0
grad_input = np.multiply(grad_input, mask)      # 填充区域的梯度不计入计算
grad_output = np.zeros((num_rois, num_channels, height, weight))        # 反向传播的输出
for n in range(num_rois):
    # 求bin的weight和height
    roi_start_w = np.round(rois[n, 1])
    roi_start_h = np.round(rois[n, 2])

    rois_weight = np.max([rois[n,3] - rois[n,1], 1])                     # 每个区域的宽度
    rois_height = np.max([rois[n,4] - rois[n,2], 1])                     # 每个区域的高度

    pooled_weight = np.ceil(np.float(rois_weight) / rois_height * pooled_height)        # 和rois等比例的池化
    pooled_weight = int(pooled_weight)

    bin_size_w = np.float(rois_weight) / pooled_weight                  # 每个bin块的宽度
    bin_size_h = np.float(rois_height) / pooled_height                  # 每个bin块的高度
    for c in range(num_channels):
        for ph in range(pooled_height):
            for pw in range(pooled_weight):

                hstart = np.floor(ph * bin_size_h)
                wstart = np.floor(pw * bin_size_w)
                hend = np.ceil((ph + 1) * bin_size_h)
                wend = np.ceil((pw + 1) * bin_size_w)

                hstart = min(max(hstart + roi_start_h, 0),height)           # 将每个bin限制在图片的尺寸范围内
                hend = min(max(hend + roi_start_h, 0), height)
                wstart = min(max(wstart + roi_start_w, 0), weight)
                wend = min(max(wend + roi_start_w, 0), weight)
                
                hstart, hend, wstart, wend = map(lambda x: int(x), [hstart, hend, wstart, wend])

                temp = np.zeros((hend-hstart, wend-wstart))             # 每个bin区域的临时值
                temp = temp.flatten()
                temp[int(argmax[n,c,ph,pw])] = grad_input[n,c,ph,pw]
                temp = np.reshape(temp, (hend-hstart, -1))
                grad_output[n,c, hstart:hend, wstart:wend] = temp           # 对一块bin进行赋值

grad_output = np.sum(grad_output, axis=0)
temp = 0


