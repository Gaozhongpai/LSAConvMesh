#%%
import matplotlib
import matplotlib.pyplot as plt 
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 12
# font = {'family' : 'Times New Roman', 'size': 10}
# matplotlib.rc('font', **font)
x = [3.48, 4.32, 3.42, 3.36 ,5.22]
# x = [53, 61, 62, 64 ,90]
y = [5.258, 5.238, 3.492, 4.46, 6.096]
s = [446, 361, 2478, 447, 441]
n = ["Spiral", "COMA", "LSA-Conv (ours)", "LSA-Tiny (ours)", "FeaStNet"]
scatter = plt.scatter(x, y, s=s)

plt.annotate(n[0], # this is the text
                (x[0],y[0]), # this is the point to label
                textcoords="offset points", # how to position the text
                xytext=(0,15), # distance from text to points (x,y)
                ha='center') # horizontal alignment can be left, right or center
plt.annotate(n[1], # this is the text
                (x[1],y[1]), # this is the point to label
                textcoords="offset points", # how to position the text
                xytext=(0,15), # distance from text to points (x,y)
                ha='center') # horizontal alignment can be left, right or center
plt.annotate(n[2], # this is the text
                (x[2],y[2]), # this is the point to label
                textcoords="offset points", # how to position the text
                xytext=(20,30), # distance from text to points (x,y)
                ha='center') # horizontal alignment can be left, right or center
plt.annotate(n[3], # this is the text
                (x[3],y[3]), # this is the point to label
                textcoords="offset points", # how to position the text
                xytext=(28,15), # distance from text to points (x,y)
                ha='center') # horizontal alignment can be left, right or center
plt.annotate(n[4], # this is the text
                (x[4],y[4]), # this is the point to label
                textcoords="offset points", # how to position the text
                xytext=(-10,-20), # distance from text to points (x,y)
                ha='center') # horizontal alignment can be left, right or center
plt.xlabel("Time of inferring test set (s)")
plt.ylabel("L2 errors (mm)")

# produce a legend with a cross section of sizes from the scatter
handles, labels = scatter.legend_elements(prop="sizes", alpha=0.4)
legend2 = plt.legend(handles, labels, loc="lower right", title="Parameter # (K)")

plt.show()


print(x)

# %%
