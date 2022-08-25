import numpy as np
import torch
from torch import nn
from torch.nn.functional import normalize
from scipy.stats import rankdata
from tqdm import tqdm
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import pandas as pd
import time
import multiprocessing as mp
import seaborn as sns


def get_parser():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="Language transfer", fromfile_prefix_chars='@')

    parser.add_argument("--embedding_dir", type=str, default=None,
                        help="folder where the embedding file stored")

    parser.add_argument("--output_dir", type=str, default=None,
                        help="folder where to save the output")

    parser.add_argument("--pairs_file", type=str, default=None,
                        help="npy file which contains the file name of pairs") 

    parser.add_argument("--bin_width", type=int, default=10,
                        help="bin width for the hist plot")

    parser.add_argument("--corpus_split", type=int, default=0,
                        help="split the corpus when calculating cos score for limited GPU memory")
               

    return parser

def get_tensor(pt_path, bs):
    t = torch.load(pt_path)
    if t.shape[0] < bs:
        res = torch.zeros(bs,t.shape[1],t.shape[2])
        res[:t.shape[0]] = t # End Pad
    else:
        res = t

    res = res[:bs].flatten().clone()
    res_norm = normalize(res,p=2.0,dim=0)
    return res_norm
    

def get_distance(pt_folder, cur_p, bs):
    try:
        e_1 = get_tensor(pt_folder / (str(cur_p[0]) +'.pt'), bs)
        e_2 = get_tensor(pt_folder / (str(cur_p[1]) +'.pt'), bs)
        return e_2-e_1
    except:
        print('In dist_func')
        exit(0)

def get_sum(pt_folder, pairs, bs, num_cpu):
    nprocs = num_cpu if num_cpu else 1
    t1 = time.time()

    temp = []
    for p1 in pairs.keys():
        temp.extend([(p1,p2) for p2 in pairs[p1]])

    if num_cpu :
        with mp.Pool(processes=nprocs) as p:
            emb_all = p.starmap(get_distance, [(pt_folder, p, bs) for p in temp])
    else:
        emb_all = [get_distance(pt_folder, p, bs) for p in temp]
    
    
    emb_sum = [x.numpy() for x in emb_all]
    emb_sum = np.array(emb_sum)
    emb_sum = np.sum(emb_sum,axis=0)

    t4 = time.time()

    print(f"PT : Total time sum calculation {(t4-t1) :.2f}s with {nprocs} CPUs")
    print(emb_sum.shape)

    return torch.tensor(emb_sum)

def get_cos_score(p_emb,p_x,p_counter,p_rand,cos):

    output = cos(p_emb, p_x).item()
    output_rand = cos(p_emb, p_rand).item()
    output_counter = cos(p_emb, p_counter).cpu().item()

    return [output,output_counter,output_rand]

def get_cos_score_gpu(p_emb,p_x,p_counter,p_rand,cos):

    output = cos(p_emb, p_x).cpu().item()
    output_rand = cos(p_emb, p_rand).cpu().item()
    output_counter = cos(p_emb, p_counter).cpu().item()

    return [output,output_counter,output_rand]

if __name__ == "__main__":

    parser_initial = get_parser()
    params = parser_initial.parse_args()

    if params.embedding_dir:
        pt_folder = Path(params.embedding_dir)
    else:     
        pt_folder = Path(params.output_dir) / "embeddings_pt/"
    pt_files = [x for x in pt_folder.glob('*.pt')]
    print(f"\nEmbeddings folder : {pt_folder}")
    print(f"Output folder : {params.output_dir}")

    num_files = len(pt_files)
    print(f"Number of files : {num_files}")

    pairs = np.load(params.pairs_file, allow_pickle=True).item()
    num_pairs = np.sum([len(x) for x in pairs.values()])
    print(f"Number of pairs : {num_pairs}")
    

    input_shape = torch.load(pt_files[0]).shape
    print(f"Input Shape : {input_shape}")

    bs = input_shape[0]

    actual_bptt = bs * input_shape[1] * input_shape[2]
    actual_bptt_str = f"{bs}x{input_shape[1]}x{input_shape[2]}"


    print(f"Actual BPTT : {actual_bptt_str} = {actual_bptt}")

    output_path = Path(params.output_dir)
    output_path.mkdir(exist_ok=True,parents=True)

    cur_relationship = params.pairs_file.split("/")[-1].split(".")[0]
    r_save_folder = Path(params.output_dir) / "R_vector" / cur_relationship
    r_save_folder.mkdir(exist_ok=True,parents=True)

    e_sum = get_sum(pt_folder, pairs, bs, 32)


    res_dic = {
        "EmbB1" : [],
        "EmbB2" : [],
        "Rank" : [],
        "Score" : [],
        "Emb_Type" : [],
    }

    use_gpu = True
    nprocs = 32

    t1 = time.time()
    all_embs_name = np.arange(num_files)
    if use_gpu:
        with mp.Pool(processes=nprocs) as p:
            all_embs_tensor = p.starmap(get_tensor, [(pt_folder / (str(emb) +'.pt'), bs) for emb in all_embs_name])
    else:
        pool = mp.Pool(processes=nprocs)
        all_embs_tensor = pool.starmap(get_tensor, [(pt_folder / (str(emb) +'.pt'), bs) for emb in all_embs_name])
    t2 = time.time()
    print(f"All embeddings loaded in {t2-t1 :.2f}s")

    if use_gpu:
        gpu_nprocs = 8
        ctx = torch.multiprocessing.get_context("spawn")
        gpu_pool = ctx.Pool(gpu_nprocs)

    cos = nn.CosineSimilarity(dim=0,eps=1e-6)
    if use_gpu:
            cos = cos.cuda()

    key_list = [x for x in pairs.keys()]
    value_list = [x for x in pairs.values()]

    diff_r_list = []
    file_progress = tqdm(range(len(pairs.keys())),"Files")
    for i in file_progress:
        cur_p1 = key_list[i]
        cur_value = value_list[i]
        e_c = all_embs_tensor[int(cur_p1)]
        e_d_list = [all_embs_tensor[int(x)] for x in cur_value]

        e_x = (e_sum - (torch.sum(torch.cat([x.unsqueeze(0) for x in e_d_list]),axis=0) - len(e_d_list) * e_c)) / (num_pairs - len(e_d_list))
        e_x = normalize(e_x,p=2.0,dim=0)         

        random_embeddings = all_embs_name
        random_embeddings_tensor = all_embs_tensor
        
        t1 = time.time()

        p_x = e_c + e_x
        p_counter = e_c - e_x
        p_rand = e_c + torch.randn_like(e_x)

        if use_gpu:
            p_x = p_x.cuda()
            p_counter = p_counter.cuda()
            p_rand = p_rand.cuda()
            split = params.corpus_split
            if split:
                cosine_similarities = []
                interval = int(len(random_embeddings_tensor) / split)
                for k in range(split):
                    statr_pos = k * interval
                    if k == split-1:
                        temp_tensor = random_embeddings_tensor[statr_pos:]
                    else:
                        end_pos = (k+1) * interval
                        temp_tensor = random_embeddings_tensor[statr_pos:end_pos]
                    temp = gpu_pool.starmap(get_cos_score_gpu, [(e_emb.cuda(),p_x,p_counter,p_rand,cos) for e_emb in temp_tensor])
                    cosine_similarities.extend(temp)
                    del temp_tensor
                    torch.cuda.empty_cache()
            else:
                cosine_similarities = gpu_pool.starmap(get_cos_score_gpu, [(e_emb.cuda(),p_x,p_counter,p_rand,cos) for e_emb in random_embeddings_tensor])
        else:
            cosine_similarities = pool.starmap(get_cos_score, [(e_emb,p_x,p_counter,p_rand,cos) for e_emb in random_embeddings_tensor])

        t2 = time.time() 

        pfx_dic = {
            "Last.Cos" : f"{t2 - t1 :.2f}s",
            "Pair" : f"{i+1}/{len(pairs)}"
            }

        file_progress.set_postfix(pfx_dic)

        # higher the cosine, the higher the similarity
        cosine_similarities = np.array(cosine_similarities)

        for l,r in enumerate(["Original","Negative","Random"]):
            cur_res = cosine_similarities[:,l].tolist()
            ranks = rankdata(cur_res, method='min')
            ranks = np.max(ranks)+1 - ranks
            rank_list = []
            for cur_p2 in cur_value:
                index = np.where(random_embeddings == int(cur_p2))[0][0]
                rank_list.append(ranks[index])
                res_dic["Rank"].append(ranks[index])
                res_dic["Score"].append(cur_res[index])
            for cur_p2 in cur_value:
                res_dic["Emb_Type"].append(r)
                res_dic["EmbB1"].append(cur_p1)
                res_dic["EmbB2"].append(cur_p2)
            

    # CSV file for the experiment data
    cur_relationship = params.pairs_file.split("/")[-1].split(".")[0]
    csv_save_name = f"bptt{actual_bptt_str}.csv"
    csv_folder = Path(params.output_dir) / "csvs" / cur_relationship
    csv_folder.mkdir(exist_ok=True,parents=True)
    df_res = pd.DataFrame(res_dic)
    df_res.to_csv(csv_folder / csv_save_name)

    # Histplot for the rank distribution
    save_name = f"_bptt{actual_bptt_str}.jpg"

    hist_save_name = "hist_rank_" + save_name
    plt.figure(figsize=(20,10))
    sns.histplot(
        data=df_res
        ,x="Rank"
        ,hue="Emb_Type"
        ,multiple="dodge"
        ,binwidth=params.bin_width
        ,stat='probability'
        ,common_norm = False
        )
    plt.xlabel('Ranks',fontsize=15)
    plt.ylabel('Frequency',fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    title = f"Histogram of Rank (BPTT {actual_bptt_str})"
    plt.title(title,fontsize=20)
    plot_path = Path(params.output_dir) / "plots" / cur_relationship
    plot_path.mkdir(parents=True,exist_ok=True)
    plt.tight_layout()
    plt.savefig(plot_path / hist_save_name)

    # Scatter Plot for the pairwise distribution
    scatter_save_name = "scatter_point_" + save_name
    df_emb = df_res[["Rank","EmbB1","EmbB2","Emb_Type"]]
    df_emb = df_emb[df_emb["Emb_Type"] == "Original"]
    df_emb = df_emb.drop_duplicates()
    df_emb = df_emb.sort_values(["EmbB1","Rank"])

    rank = df_emb["Rank"].tolist()
    b1 = [f"{x}" for x in df_emb["EmbB1"].tolist()]
    b2 = [f"{x}" for x in df_emb["EmbB2"].tolist()]

    df_res =pd.DataFrame(
        {
            "Rank" : rank,
            "b1" : b1
        }
    )

    plt.figure(figsize=(50,50))
    sns.set_style("darkgrid", {"axes.facecolor": ".6"})
    ax = sns.scatterplot(
            data=df_res
            ,y="b1"
            ,x="Rank"
            ,s=500
            )

    lastb1 = ""        
    for i, txt in enumerate(b2):
        if b1[i] == lastb1:
            prefix = "      " + prefix
        else:
            prefix = "  "
        ax.annotate(prefix+txt, (rank[i], b1[i]),fontsize=20)
        lastb1 = b1[i]
        
    plt.ylabel('Embs',fontsize=15)
    plt.xlabel('Rank',fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    p_title = f"Scatter Plot for the pairwise distribution- (BPTT {actual_bptt_str})"
    plt.title(p_title,fontsize=20)
    plt.title(p_title,fontsize=20)
    plt.tight_layout()
    plt.savefig(plot_path / scatter_save_name)