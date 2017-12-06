
vowels = {'AO':'O', 'AA':'a', 'IY':'i', 'IH':'I', 'EY':'e I', 'EH':'E',
        'AE':'ae', 'OW':'o U', 'UH':'U', 'UW':'u', 'ER':'E R', 'AH':'2', 
        'AY':'E I', 'AW':'E U', 'OY':'U I', 'AX':'E'}

consonants = {'P':'p', 'T':'t', 'K':'k', 'B':'b', 'D':'d', 
        'G':'g', 'M':'m', 'N':'n', 'NG':'N', 'F':'f', 'V':'v', 
        'TH':'T', 'DH':'D', 'S':'s', 'Z':'z', 'SH':'S', 'ZH':'Z', 
        'W':'w', 'R':'r', 'L':'l', 'Y':'j', 'HH':'h', 'CH':'tS', 
        'JH':'dZ'}


data = open('words_ipa.txt', 'r')
output = open('words_converted.txt', 'w')

for line in data:
    line.strip()
    if line:
        line = line.split()
        if '(4)' in line[0] or '(3)' in line[0] or '(2)' in line[0]:
            continue 

        tmp = ['\t']
        short = line[1:]
        for sound in short:
            if sound in vowels:
                tmp.append(vowels[sound])
            elif sound in consonants:
                tmp.append(consonants[sound])
            else:
                print "uh oh", sound

        tmp1 = [line[0]]
        tmp1.append('\t')

        tmp1.extend(short)
        tmp1.extend(tmp)
        line = tmp1

        for item in line:
            if item == '\t':
                output.write(item)
            else:
                output.write(item+" ")
        output.write('\n')

data.close()
output.close()

