import pandas as pd
import matplotlib.pyplot as plt

file_path = "data/VRSA-FR/SSQ.xlsx"
df = pd.read_excel(file_path, header=None)
df.drop(0, axis=1, inplace=True)
df.columns = df.columns-1

ignored_datas = ((1,9), (3,5), (3,6), (4,0), (4,3), (4,15), (5,5), (5,9), (6,12), (6,18), (7,1), (7,7), (9,6), (10,6), (13,0), (13, 2), (13,10), (14,0), (14,10), (14,17), (15,1), (22,14), (23,2))

selected_indices = pd.DataFrame([[0]*4]*25)
selected_values = pd.DataFrame([[0.0]*4]*25)

for subject in df.T:
    for scene_index in range(0, 20, 2):
        if (subject, scene_index) in ignored_datas or (subject, scene_index+1) in ignored_datas:
            continue
        scene = df.iloc[subject, scene_index], df.iloc[subject, scene_index+1]
        if selected_values.iloc[subject, 0] == 0 and df.iloc[subject, scene_index+2:].isin([scene[0]]).any() and df.iloc[subject, scene_index+2:].isin([scene[1]]).any() and scene[0] != scene[1]:
            selected_indices.iloc[subject, 0] = scene_index
            selected_indices.iloc[subject, 1] = scene_index+1
            selected_values.iloc[subject, 0] = df.iloc[subject, scene_index]
            selected_values.iloc[subject, 1] = df.iloc[subject, scene_index+1]
        elif selected_values.iloc[subject, 0] != 0 and df.iloc[subject, scene_index+2:].isin([scene[0]]).any() and df.iloc[subject, scene_index+2:].isin([scene[1]]).any() and not df.iloc[subject, :scene_index].isin([scene[0]]).any() and not df.iloc[subject, :scene_index].isin([scene[1]]).any() and scene[0] != scene[1]:
            selected_indices.iloc[subject, 2] = scene_index
            selected_indices.iloc[subject, 3] = scene_index+1
            selected_values.iloc[subject, 2] = df.iloc[subject, scene_index]
            selected_values.iloc[subject, 3] = df.iloc[subject, scene_index+1]
            
        if selected_values.iloc[subject, 0] != 0 and selected_values.iloc[subject, 2] != 0 :
            break

    if selected_values.iloc[subject, 0] != 0 and selected_values.iloc[subject, 2] == 0 :
        for scene_index in range(18, -1, -2):
            if (subject, scene_index) in ignored_datas or (subject, scene_index+1) in ignored_datas:
                continue
            scene = df.iloc[subject, scene_index], df.iloc[subject, scene_index+1]
            if selected_values.iloc[subject, 0] != 0 and df.iloc[subject, scene_index+2:].isin([scene[0]]).any() and df.iloc[subject, scene_index+2:].isin([scene[1]]).any() and scene[0] != scene[1] and scene_index != selected_indices.iloc[subject, 0]:
                selected_indices.iloc[subject, 2] = scene_index
                selected_indices.iloc[subject, 3] = scene_index+1
                selected_values.iloc[subject, 2] = df.iloc[subject, scene_index]
                selected_values.iloc[subject, 3] = df.iloc[subject, scene_index+1]
            elif selected_values.iloc[subject, 0] != 0 and (df.iloc[subject, scene_index+2:].isin([scene[0]]).any() or df.iloc[subject, scene_index+2:].isin([scene[1]]).any()) and not df.iloc[subject, :scene_index].isin([scene[0]]).any() and not df.iloc[subject, :scene_index].isin([scene[1]]).any() and scene[0] != scene[1] and scene_index != selected_indices.iloc[subject, 0]:
                selected_indices.iloc[subject, 2] = scene_index
                selected_indices.iloc[subject, 3] = scene_index+1
                selected_values.iloc[subject, 2] = df.iloc[subject, scene_index]
                selected_values.iloc[subject, 3] = df.iloc[subject, scene_index+1]
            if selected_values.iloc[subject, 0] != 0 and selected_values.iloc[subject, 2] != 0 :
                break

    if selected_values.iloc[subject, 0] == 0 and selected_values.iloc[subject, 2] == 0 :
        for scene_index in range(18, -1, -2):
            if (subject, scene_index) in ignored_datas or (subject, scene_index+1) in ignored_datas:
                continue
            scene = df.iloc[subject, scene_index], df.iloc[subject, scene_index+1]
            if selected_values.iloc[subject, 0] == 0 and (df.iloc[subject, scene_index+2:].isin([scene[0]]).any() or df.iloc[subject, scene_index+2:].isin([scene[1]]).any()) and scene[0] != scene[1]:
                selected_indices.iloc[subject, 0] = scene_index
                selected_indices.iloc[subject, 1] = scene_index+1
                selected_values.iloc[subject, 0] = df.iloc[subject, scene_index]
                selected_values.iloc[subject, 1] = df.iloc[subject, scene_index+1]
            elif selected_values.iloc[subject, 0] != 0 and (df.iloc[subject, scene_index+2:].isin([scene[0]]).any() or df.iloc[subject, scene_index+2:].isin([scene[1]]).any()) and not df.iloc[subject, :scene_index].isin([scene[0]]).any() and not df.iloc[subject, :scene_index].isin([scene[1]]).any() and scene[0] != scene[1]:
                selected_indices.iloc[subject, 2] = scene_index
                selected_indices.iloc[subject, 3] = scene_index+1
                selected_values.iloc[subject, 2] = df.iloc[subject, scene_index]
                selected_values.iloc[subject, 3] = df.iloc[subject, scene_index+1]

msk_df = pd.DataFrame([[False]*20]*25)
for subject in selected_indices.T:
    for index in selected_indices.loc[subject]:
        if msk_df.iloc[subject, index] == True:
            msk_df.iloc[subject, index] = False
        elif msk_df.iloc[subject, index] == False:
            msk_df.iloc[subject, index] = True

for i in selected_indices.values:
    print(", ".join(map(str, set(i))))

a = pd.concat([df[msk_df][i] for i in range(df[msk_df].shape[1])]).value_counts().sort_index()
a
plt.subplot(1,2,1)
plt.bar(a.index, a.values)

plt.xlabel("Values")
plt.ylabel("frequence")
plt.title("Orig. Test Hist.")

a = pd.concat([df[~msk_df][i] for i in range(df[~msk_df].shape[1])]).value_counts().sort_index()
a
plt.subplot(1,2,2)
plt.bar(a.index, a.values)

plt.xlabel("Values")
plt.ylabel("frequence")
plt.title("Orig. Train Hist.")
plt.show()
