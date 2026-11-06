import numpy as np
S = "<project>/runs/seed0/loop2/round4/agent"
W = dict(np.load(S + "/working.npz")); p = 97
w1, b1, w2, b2 = W["fc1.weight"], W["fc1.bias"], W["fc2.weight"], W["fc2.bias"]
d = np.load(S + "/train_pairs.npy"); A, B, Y = d.T
H = np.maximum(0, w1[:, A].T + w1[:, p + B].T + b1)
L = H @ w2.T + b2
mean_logit = L.mean(0)  # per class over train pairs
print("class offset (mean logit) std incl b2:", mean_logit.std(), " w/o b2:", (H @ w2.T).mean(0).std(), " b2 std", b2.std())
print("max/min offset", mean_logit.max() - mean_logit.mean(), mean_logit.min() - mean_logit.mean())
# correct-class logit vs others
corr = L[np.arange(len(Y)), Y]
print("mean correct logit - mean logit", (corr - L.mean(1)).mean())
# predicted-class frequency on train vs offset
pred = L.argmax(1)
# correlation of per-class train acc with offset
acc_c = np.array([(pred[Y == c] == c).mean() for c in range(p)])
print("corr(offset, class acc)", np.corrcoef(mean_logit, acc_c)[0, 1])
# softmax probs: mean predicted prob per class vs 1/p
P = np.exp(L - L.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
print("mean p_correct", P[np.arange(len(Y)), Y].mean())
np.save(S + "/scratch/offset.npy", mean_logit - mean_logit.mean())
