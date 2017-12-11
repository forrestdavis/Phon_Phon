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


def find_sound(sound):

    feats = load_features()

    if sound == "R":
        sound = 'r'

    #Binarize the feature values for each sound
    #append them to line for output
    feat = feats[sound]
    tmp = []
    for value in features:
        #has feat
        if '+'+value in feat:
            tmp.append(1)
        #Has unary feat
        elif '?'+value in feat:
            tmp.append(1)
        #Doesn't have feat
        elif '-'+value in feat:
            tmp.append(-1)
        #Feature doesn't apply gets 0
        else:
            tmp.append(0)
    return tmp

if __name__ == "__main__":

    sibilants = ['s', 'z', 'S', 'Z', 'tS', 'dZ']
    voiceless = ['p', 't', 'k', 'f', 'T']

    values = []

    for sound in sibilants:
        values.append(find_sound(sound))
        #print sound, find_sound(sound)

    print values
    tmp = []

    first = values[0]

    for x in range(len(first)):
        not_in = 0
        for value in values:
            if first[x] != value[x]:
                not_in = 1
        if not not_in:
            tmp.append((x, first[x]))
    print tmp

    best_sound = ''
    total_values = 0
    for sound in voiceless:
        number_common = 0
        values = find_sound(sound)
        for pair in tmp:
            if values[pair[0]] == pair[1]:
                number_common += 1
        if total_values <= number_common:
            total_values = number_common
            best_sound = sound
        print sound, number_common

    print best_sound
    print total_values



