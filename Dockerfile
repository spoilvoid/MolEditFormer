FROM nvcr.io/nvidia/pytorch:22.01-py3 as base

#create a new new user
RUN useradd -ms /bin/bash zengyu

#change to this user
# USER zengyu

#set working directory
WORKDIR /home/zengyu

RUN chmod -R 777 /home/zengyu
RUN chmod -R 777 /usr/bin
RUN chmod -R 777 /bin
RUN chmod -R 777 /usr/local
RUN chmod -R 777 /opt/conda

RUN conda install -y python=3.7

RUN pip install rdkit
RUN conda install -y -c conda-forge -c pytorch pytorch=1.9.1
RUN conda install -y -c pyg -c conda-forge pyg==2.0.3

RUN pip install requests
RUN pip install tqdm
RUN pip install matplotlib
RUN pip install spacy
RUN pip install ipykernel
RUN pip install notebook

# for SciBert
RUN pip install boto3
RUN pip install transformers

# for MoleculeNet
RUN pip install ogb==1.2.0

# install pysmilesutils
RUN python -m pip install git+https://github.com/MolecularAI/pysmilesutils.git

RUN pip install deepspeed

# install Megatron
RUN cd /tmp && git clone https://github.com/MolecularAI/MolBART.git --branch megatron-molbart-with-zinc && cd /tmp/MolBART/megatron_molbart/Megatron-LM-v1.1.5-3D_parallelism && pip install .

# install apex
RUN cd /tmp && git clone https://github.com/chao1224/apex.git
RUN cd /tmp/apex/ && pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]
#expose port for Jupyter
EXPOSE 8888