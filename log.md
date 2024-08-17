# 从视频中提取3D模型
首先运行 infer_mead.py  
然后 utils/combine_npz.py ，将每一段视频的参数合并成一个文件  
再运行 param2mesh.py ，将 param 提取为npy (T, 5023, 3)  
