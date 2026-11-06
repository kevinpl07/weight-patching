import sys, json
sys.argv=['x','68']
exec(open('fold.py').read().split("print('with fold'")[0])
ops.append({"tensor":"fc2.bias","rows":list(range(p)),"op":"set","value":[float(v) for v in Wf['fc2.bias']]})
json.dump(ops,open('<project>/runs/seed0/loop/round4/agent/scratch/edit_final.json','w'))
print(len(ops),'ops; expected train',ev(Wf))
