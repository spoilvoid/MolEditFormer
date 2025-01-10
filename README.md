# MolEditFormer 
**Please run the given shell scripts in the root folder.**  
MoleculeSTM在文本处理部分的text为一个列表，其中每一个元素来源于某个database  
在encode text的过程中，MoleculeSTM输入了一个batch，每个元素为一个描述list  
在组织成为Dataset的过程中，List中的每一个元素都成为一个单独的样本，所以在MoleculeSTM中会存在多个相同CID的数据  
实际在模型中每一个text样本都是一段文本不变  





