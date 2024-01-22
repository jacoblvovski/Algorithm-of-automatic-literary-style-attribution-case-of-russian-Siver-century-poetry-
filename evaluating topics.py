import json
import gensim
import zipfile

f = open('keywords.json', encoding='UTF-8')

poems = json.load(f)
topics_dict = {}

model_url = 'http://vectors.nlpl.eu/repository/20/180.zip'
model_file = model_url.split('/')[-1]
with zipfile.ZipFile(model_file, 'r') as archive:
   stream = archive.open('model.bin')
   model = gensim.models.KeyedVectors.load_word2vec_format(stream, binary=True)

f1 = open('Futurists processed for RusVectores.txt', encoding='utf-8')

topics_futurism = f1.readlines()

for topic in topics_futurism:
    topics_dict[topic] = 'Футуризм'
topics_futurism = [topic.split() for topic in topics_futurism]

f2 = open('Symbolists processed for RusVectores.txt', encoding='utf-8')

topics_symbolism = f2.readlines()

for topic in topics_symbolism:
    topics_dict[topic] = 'Символизм'

topics_symbolism = [topic.split() for topic in topics_symbolism]

f3 = open('Akmeists processed for RusVectores.txt', encoding='utf-8')

topics_akmeism = f3.readlines()

for topic in topics_akmeism:
    topics_dict[topic] = 'Акмеизм'

topics_akmeism = [topic.split() for topic in topics_akmeism]



def topic_similarity(poem,topic):
    keywords = poem['keywords'].split()
    topic = topic.split()
    print(topic)
    print(keywords)
    similarity_scores = []
    missed = 0
    for word in topic:
        for keyword in keywords:
            try:
                print(word,keyword)
                score = model.similarity(word, keyword)
                print(score)
                similarity_scores.append(score)
            except:
                missed +=1
    print('Пропущено', missed)
    avg_score = sum(similarity_scores)/len(similarity_scores)
    return avg_score

topic_scores_dict = {}

for topic in topics_dict.keys():
    scores = []
    for poem in poems:
        if poem['style'] == topics_dict[topic]:
            scores.append(topic_similarity(poem,topic))
    avg_score = sum(scores)/len(scores)
    topic_scores_dict[topic] = avg_score

sorted_topic_scores = sorted(topic_scores_dict.items(), key=lambda x:x[1], reverse=True)

sorted_topic_scores_with_style = [topic + (topics_dict[topic[0]],) for topic in sorted_topic_scores]

print(*sorted_topic_scores_with_style, sep='\n')
