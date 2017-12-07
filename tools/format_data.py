import os
import sys

#Feature order for data
features = ['High', 'Back', 'Low', 'ATR', 'Round', 'Syllabic', 'Consonantal',
        'Sonorant', 'Continuant', 'Nasal', 'Lateral', 'DR', 'Voice', 
        'Labial', 'Coronal', 'Dorsal', 'Laryngeal', 'Anterior',
        'Distributive', 'Strident']

def load_features(feat_file = "../features"): 

    data = open(feat_file, 'r')
    feats = {}


    for line in data:
        line = line.strip()
        if line:
            if line[0] == "#":
                continue 
            line = line.split()

            if line[0] not in feats:
                feats[line[0]] = line[1:]
            else:
                print "uh oh"

    data.close()

    return feats


def format_data(data_file = "../data/data.txt"):

    feats = load_features()

    data = open(data_file, 'r')
    output = open('../data/output', 'w')

    
    for line in data:
        line = line.strip()
        if line:
            line = line.split('\t\t')
            item = line[1].split()
            re_line = line[:2]
            for sound in item:
                #Correct error in previous script
                #:(
                if sound == "R":
                    sound = 'r'

                #Binarize the feature values for each sound
                #append them to line for output
                feat = feats[sound]
                tmp = []
                for value in features:
                    #has feat
                    if '+'+value in feat:
                        tmp.append('1')
                    #Has unary feat
                    elif '?'+value in feat:
                        tmp.append('1')
                    #Doesn't have feat
                    elif '-'+value in feat:
                        tmp.append('-1')
                    #Feature doesn't apply gets 0
                    else:
                        tmp.append('0')
                tmp = ' '.join(tmp)
                re_line.append(tmp)
            re_line.append(line[2])
            line = re_line

            for x in range(len(line)):
                if x == len(line)-1:
                    output.write(line[x]+'\n')
                else:
                    output.write(line[x]+'\t\t')

    data.close()
    output.close()

    os.system("sh random.sh")

if __name__ == "__main__":

    format_data()
