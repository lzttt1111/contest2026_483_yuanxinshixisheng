"""Monotone scalar calibration with held-out acceptance; rejected fits are never used."""
from bisect import bisect_left,bisect_right
import math

def isotonic_fit(pairs):
    grouped={}
    for x,y in pairs:grouped.setdefault(x,[]).append(y)
    xs=sorted(grouped)
    blocks=[]
    for x in xs:
        ys=grouped[x];blocks.append([x,x,len(ys),sum(ys)])
        while len(blocks)>1 and blocks[-2][3]/blocks[-2][2]>blocks[-1][3]/blocks[-1][2]:
            right=blocks.pop();left=blocks.pop()
            blocks.append([left[0],right[1],left[2]+right[2],left[3]+right[3]])
    knots=[]
    for lo,hi,n,total in blocks:
        knots.append([lo,total/n])
        if hi!=lo:knots.append([hi,total/n])
    return knots

def predict(knots,x):
    xs=[k[0] for k in knots]
    i=bisect_left(xs,x)
    if i==0:return knots[0][1]
    if i==len(knots):return knots[-1][1]
    left,right=knots[i-1],knots[i]
    return left[1]+(right[1]-left[1])*(x-left[0])/(right[0]-left[0])

def quantile(values,q):
    a=sorted(values);return a[round((len(a)-1)*q)]

def percentile(values,x):
    return (bisect_left(values,x)+bisect_right(values,x))*50/len(values)

def validate(train,test,reference):
    if len(train)<200 or len(test)<50 or len({x for x,_ in train})<20:
        return {"accepted":False,"reason":"insufficient_independent_pairs","train_n":len(train),"test_n":len(test)}
    knots=isotonic_fit(train)
    baseline=[abs(x-y) for x,y in test]
    errors=[abs(predict(knots,x)-y) for x,y in test]
    score_errors=[abs(percentile(reference,predict(knots,x))-percentile(reference,y)) for x,y in test]
    mae=sum(errors)/len(errors);old_mae=sum(baseline)/len(baseline)
    p95=quantile(errors,.95);old_p95=quantile(baseline,.95)
    score_mae=sum(score_errors)/len(score_errors);score_p95=quantile(score_errors,.95)
    accepted=mae<old_mae and p95<=old_p95 and score_mae<=5 and score_p95<=10
    return {"accepted":accepted,"method":"pava_linear_interpolation","knots":knots if accepted else [],
            "train_n":len(train),"test_n":len(test),"identity_mae":old_mae,"mapped_mae":mae,
            "identity_p95":old_p95,"mapped_p95":p95,"score_mae":score_mae,"score_p95":score_p95,
            "acceptance_limits":{"score_mae":5,"score_p95":10},
            "reason":None if accepted else "held_out_mapping_gate_failed"}

