# Algorithm-of-automatic-literary-style-attribution-case-of-russian-Siver-century-poetry
В этом репозитории хранится код разработанного мной алгоритма для автоматического определения направления поэтических текстов Серебряного века, а также данные использованные для создания и тестирования алгоритма. <br/>
В файле "Выборка.xlsx" хранится эксель-таблица с стихотворениями, которые анализировались в исследовании. <br/>
В файле "algorithm 2_updated version.py" хранится код алгоритма на языке Python (к сожалению ограничения Github не позволяют загрузить в репозиторий скачанную модель Rusvectores, однако её можно скачать используя VPN по ссылке http://vectors.nlpl.eu/repository/20/180.zip). <br/>
В файле "log_reg.py" находится код, с помощью которого обучалась используемая в исследовании модель наивного байесовского классификатора. <br/>
В файлах "predictions.json", "predictions_without_hlebnikon.json", "predictions_without_kruchenyh.json" находятся предсказанные алгоритмом направления для всего корпуса, корпуса без учёта текстов Хлебникова, корпуса без учёта текстов Кручёных. <br/>
В файлах "топики акмеизм.txt", "топики символизм.txt", "топики футуризм.txt" находятся выделенные моделью LDA темы из текстов акмеистов, символистов и футуристов соответственно. <br/>
В файле "evaluating_keyword-approach.py" находится код, который использовался для оценки работы алгоритма. <br/>
В файле "evaluating topics.py" находится код, который использовался для составления "рейтинга успешности тем". <br/>
