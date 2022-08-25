import argparse
import pandas as pd

def get_parser():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="Language transfer", fromfile_prefix_chars='@')

    parser.add_argument("--positive_csv", type=str, default=None,
                        help="csv file with ranks collected using Positive input set (Q+)")

    parser.add_argument("--negative_csv", type=str, default=None,
                        help="csv file with ranks collected using Negative input set (Q-)")

    parser.add_argument("--threshold", type=int, default=462,
                        help="threshold rank for the binary classification")                     

    return parser


if __name__ == "__main__":

    parser_initial = get_parser()
    params = parser_initial.parse_args()

    df_pos = pd.read_csv(params.positive_csv)
    df_pos = df_pos[df_pos["Emb_Type"] == "Original"]

    df_neg = pd.read_csv(params.negative_csv)
    df_neg = df_neg[df_neg["Emb_Type"] == "Original"]

    tp = len(df_pos[df_pos["Rank"] < params.threshold])
    fn = len(df_pos[df_pos["Rank"] >= params.threshold])

    fp = len(df_neg[df_neg["Rank"] < params.threshold])
    tn = len(df_neg[df_neg["Rank"] >= params.threshold])


    # Precision : TP / (TP+FP) 
    precision = tp / (tp+fp)

    # Sensitivity (Recall) : TP / (TP+FN)
    recall = tp / (tp+fn)

    # Accuracy : TP + TN / (TP+TN+FN+FP)
    acc = (tp + tn) / (tp+tn+fn+fp)

    # F1-Score : 2 x Recall x Precision / (Recall + Precision) 
    f1_score = (2 * precision * recall) / (recall+precision)

    print(f"TP : {tp};  FP : {fp};  FN : {fn};  TN : {tn};  ")
    print(f'precision : {precision}')
    print(f'Recall : {recall}')
    print(f'acc : {acc}')
    print(f'f1_score : {f1_score}')