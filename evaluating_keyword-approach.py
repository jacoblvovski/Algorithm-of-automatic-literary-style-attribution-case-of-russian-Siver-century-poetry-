import json

f = open('predictions.json')

poems = json.load(f)

number_of_poems = len(poems)

true = 0
s_true = 0
f_true = 0
a_true = 0
f_general = 0
list_of_authors = []

for poem in poems:
    list_of_authors.append(poem['author'])
    if poem['style'] == 'Футуризм':
        f_general +=1

list_of_authors = list(set(list_of_authors))

author_poem_dicts = {}

for author in list_of_authors:
    author_poem_dicts[author] = list()

for poem in poems:
    author_poem_dicts[poem['author']].append(poem)

author_result_dict = {}

for author in author_poem_dicts.keys():
    author_poems = author_poem_dicts[author]
    general_number = len(author_poems)
    true = 0
    for poem in author_poems:
        if poem['style'] == poem['predicted_style']:
            true +=1
    percent = (true/general_number)*100
    author_result_dict[author] = percent



for poem in poems:
    if poem['style'] == poem['predicted_style']:
        true += 1
        if poem['style'] == 'Символизм':
            s_true +=1
        elif poem['style'] == 'Футуризм':
            f_true +=1
        elif poem['style'] == 'Акмеизм':
            a_true +=1

print(true,'из',number_of_poems, 'все.')
print(str((true/number_of_poems)*100)+'%', 'все.')
f_percent = f_true/f_general
s_percent = (s_true/131)*100
a_percent = (a_true/101)*100
print(f_true, 'из', f_general, str(f_percent)+'%', 'футуристы.')

print(s_true, 'из 131.', str(s_percent)+'%', 'символисты.')


print(a_true, 'из 101.', str(a_percent)+'%', 'акмеисты.')

for author in author_result_dict.keys():
    print(author,author_result_dict[author])

counter_sa = 0
for poem in poems:
    if poem['style'] == 'Символизм' and poem['predicted_style'] == 'Акмеизм':
        counter_sa +=1

counter_as = 0
for poem in poems:
    if poem['style'] == 'Акмеизм' and poem['predicted_style'] == 'Символизм':
        counter_as +=1

print('Символизм -> Акмеизм', counter_sa)
print('Акмеизм -> Символизм', counter_as)


counter_sf = 0

for poem in poems:
    if poem['style'] == 'Символизм' and poem['predicted_style'] == 'Футуризм':
        counter_sf +=1

print('Символизм -> Футуризм', counter_sf)

counter_fa = 0

for poem in poems:
    if poem['style'] == 'Футуризм' and poem['predicted_style'] == 'Акмеизм':
        counter_fa +=1

print('Футуризм -> Акмеизм', counter_fa)

counter_fs = 0
for poem in poems:
    if poem['style'] == 'Футуризм' and poem['predicted_style'] == 'Символизм':
        counter_fs +=1

print('Футуризм -> Символизм', counter_fs)