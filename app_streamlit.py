# Importando bibliotecas
import nltk
import streamlit as st

nltk.download('stopwords')
nltk.download('rslp')

# Título
st.write("""
Análise de Sentimento com NLTK\n
App que utiliza machine Learning para analisar emoções, opiniões e sentimentos contidos nas frases disponibilizadas.\n

""")

# Dataset

# Bases de Treinamento e de Testes
basetreinamento = {
    ('você e adorável', 'amor'),
    ('adoro a maneira como você age', 'amor'),
    ('estou apaixonado', 'amor'),
    ('meu pai esta apaixonado', 'amor'),
    ('estamos todos encantados', 'amor'),
    ('essa situação e muito doce', 'amor'),
    ('disse adeus docemente', 'amor'),
    ('tenho simpatia por aquela pessoa', 'amor'),
    ('como pode ser tão simpática!', 'amor'),
    ('que maravilhoso seu encanto', 'amor'),
    ('tenho afeição por agente como você', 'amor'),
    ('isso tudo e só felicidade', 'amor'),
    ('estou muito feliz com suas verdades', 'amor'),
    ('tão agradável', 'amor'),
    ('isso me agrada completamente', 'amor'),
    ('te agrada isso', 'amor'),
    ('estou com sentimentos maravilhosos', 'amor'),
    ('todos estão enamorados', 'amor'),
    ('foi uma paixão incrível', 'amor'),
    ('isso e muito lindo', 'amor'),
    ('seja tão gentil', 'amor'),
    ('você fez uma manobra incrível', 'amor'),
    ('sua dignidade, tem vergonha?', 'amor'),
    ('você e carinhoso com as crianças', 'amor'),
    ('que comentário amoroso', 'amor'),
    ('com tanto respeito você lida com tudo', 'amor'),
    ('sinto amor por você', 'amor'),
    ('e encantador a maneira como olha para as pessoas', 'amor'),
    ('estou disposta', 'amor'),
    ('a disposição me atacou hoje', 'amor'),
    ('acho que vou me apaixonar', 'amor'),
    ('tem muito amor lá', 'amor'),
    ('que prazer essa alegria', 'amor'),
    ('não me abandone nunca mais', 'amor'),
    ('suas atitudes estão nos alegrando', 'amor'),
    ('que beleza olha toda essa perfeição', 'amor'),
    ('como isso está belo', 'amor'),
    ('tenho emoções só de lembrar', 'amor'),
    ('me sinto emocionada com o cheiro desta comida', 'amor'),
    ('você esta iluminando a passagem de ar', 'amor'),
    ('você esta maravilhosamente bem', 'amor'),
    ('olhe que bela esta roupa', 'amor'),
    ('que atitude adorável', 'amor'),
    ('nossa como você e bonito', 'amor'),
    ('muito bom tudo isso', 'amor'),
    ('estou encantado com você', 'amor'),
    ('você cortou o meu coração', 'amor'),
    ('para que tanta felicidade?', 'amor'),
    ('esse perfume e encantador', 'amor'),
    ('ser amável não tem nada melhor', 'amor'),
    ('você e amável demais para minhas filhas', 'amor'),
    ('que cheiroso este esgoto', 'amor'),
    ('que cheiroso você esta', 'amor'),
    ('que cachorro cheiroso', 'amor'),
    ('hora que prazer', 'amor'),
    ('e prazeroso da sua parte', 'amor'),
    ('situação agradável essa', 'amor'),
    ('você só me da felicidade', 'amor'),
    ('tenho apreço a pessoas assim', 'amor'),
    ('amor e um bem da sociedade', 'amor'),
    ('que criatura adorável', 'amor'),
    ('e alegre a maneira como você vê o mundo', 'amor'),
    ('me agrada sua presença na festa', 'amor'),
    ('sinto amor por essa coisa', 'amor'),
    ('que lindo!', 'amor'),
    ('vou saborear o café agora', 'amor'),
    ('hora que garota encantadora!', 'amor'),
    ('estou apaixonada', 'amor'),
    ('isso que você disse foi muito maravilhoso', 'amor'),
    ('não seja obsceno na frente das crianças', 'amor'),
    ('não seja rude com as visitas', 'amor'),
    ('esse assunto me da prazer', 'amor'),
    ('que criança maravilhosamente travessa', 'amor'),
    ('que criança bem educada', 'amor'),
    ('estou disposta a te dar o divórcio', 'amor'),
    ('tão encantador, não tem nada mais bonito para dizer?', 'amor'),
    ('por motivo nobre, com emprego de meio amável e com possibilidade de defesa para a vítima', 'amor'),
    ('a admiração e tão nobre e honrosa que todos se atrevem a confessá-la', 'amor'),
    ('o incrível receio de ser sentimental e o mais honroso de todos os receios modernos', 'amor'),
    ('travesso gato quando fica com saudades do dono ronrona no sapato', 'amor'),
    ('isso e um ato adorável e corajoso', 'amor'),
    ('revelam apenas o que e construtivo e adorável para o povo', 'amor'),
    ('não sei como e a vida de um herói, mais a de um homem honesto e admirável', 'amor'),
    ('há coisas que temos que suportar para acharmos a vida suportável', 'amor'),
    ('as virtudes do tempo e as justiças do homem', 'amor'),
    ('gracioso e humano', 'amor'),
    ('você publicará conteúdo amoroso, respeitoso e encorajador', 'amor'),
    ('afetuoso e expressivo', 'amor'),
    ('não há animal mais elevado, inteligente, corajoso, louvável, altruísta, respeitoso e admirável do que o homem',
     'amor'),
    ('o animado debate entre políticos', 'amor'),

    ('é bom ter sua companhia', 'alegria'),
    ('contentamento é a palavra que define meu estado de espírito', 'alegria'),
    ('estou acompanhado por bons amigos', 'alegria'),
    ('estou animada', 'alegria'),
    ('ele esta todo animado', 'alegria'),
    ('tão alegres suas palavras', 'alegria'),
    ('seu amor agora é meu', 'alegria'),
    ('estou feliz', 'alegria'),
    ('isso vai me alegrar', 'alegria'),
    ('estou com muita alegria', 'alegria'),
    ('me alegra o modo como fala', 'alegria'),
    ('estou em êxtase com meu íntimo', 'alegria'),
    ('quero fazer tudo', 'alegria'),
    ('me sinto animada e cheia de vida', 'alegria'),
    ('não consigo parar de sorrir', 'alegria'),
    ('não consigo segurar a felicidade', 'alegria'),
    ('é muita alegria ter um ente querido', 'alegria'),
    ('estou realmente satisfeita', 'alegria'),
    ('acho que o karma volta, pois agora sou eu quem celebra', 'alegria'),
    ('você cumpriu suas promessas', 'alegria'),
    ('me sinto revigorada', 'alegria'),
    ('coitado está tão feliz', 'alegria'),
    ('já é tarde demais para ficar triste', 'alegria'),
    ('nosso amor está florescendo', 'alegria'),
    ('essa noite alegra só para mim', 'alegria'),
    ('eu agora estou no seu coração', 'alegria'),
    ('você mudou para melhor', 'alegria'),
    ('quando eu penso em você realmente me enche de alegria', 'alegria'),
    ('como se fosse nada você vê meu sorriso', 'alegria'),
    ('você disse carinhosamente que não se arrependeu', 'alegria'),
    ('eu sempre vou te ver', 'alegria'),
    ('ela está muito feliz', 'alegria'),
    ('a felicidade inspira as pessoas', 'alegria'),
    ('estar alegre é muito bom', 'alegria'),
    ('estou realizada e radiante depois deste dia', 'alegria'),
    ('é comovente te ver dessa maneira', 'alegria'),
    ('é comovente ver o que os filhos do Brasil conquistam', 'alegria'),
    ('como me sinto orgulhosa', 'alegria'),
    ('estou radiante', 'alegria'),
    ('a tranquilidade tomou conta de mim', 'alegria'),
    ('as pessoas gostam do meu jeito', 'alegria'),
    ('até logo passamos bons momentos juntos', 'alegria'),
    ('sinto alegria', 'alegria'),
    ('ele adorou a minha comida', 'alegria'),
    ('estou com dinheiro para a comida', 'alegria'),
    ('queria que fosse sempre o melhor dia da minha vida', 'alegria'),
    ('você está orgulhoso de mim', 'alegria'),
    ('ela aceitou a minha proposta', 'alegria'),
    ('era o meu último centavo bem gasto', 'alegria'),
    ('passei de ano na faculdade', 'alegria'),
    ('afinal você só sabe me elogiar', 'alegria'),
    ('eu fui muito elogiado', 'alegria'),
    ('é uma história muito alegre', 'alegria'),
    ('todos acreditam em mim', 'alegria'),
    ('eu sirvo para tudo mesmo', 'alegria'),
    ('ufa, faço tudo direito', 'alegria'),
    ('felicidade em dobro na minha vida', 'alegria'),
    ('fui promovida essa semana', 'alegria'),
    ('as crianças se alegram ainda mais que os adultos', 'alegria'),
    ('pra mim um dia é bom, o outro é melhor', 'alegria'),
    ('de repente ganhei o apetite', 'alegria'),
    ('oh que dia feliz', 'alegria'),
    ('estamos afundados em alegria', 'alegria'),
    ('nem um milagre pode nos tirar a felicidade', 'alegria'),
    ('só me resta aproveitar', 'alegria'),
    ('melhor que isso não pode ficar', 'alegria'),
    ('meu salário é alto', 'alegria'),
    ('passei no vestibular', 'alegria'),
    ('todos se importam comigo', 'alegria'),
    ('todos lembraram do meu aniversário', 'alegria'),
    ('tenho tanto sorte', 'alegria'),
    ('o gosto da alegria é doce', 'alegria'),
    ('sou uma mulher feliz depois de que você me deixou', 'alegria'),
    ('estou animada com a vida', 'alegria'),
    ('é um ânimo só coitadinha', 'alegria'),
    ('a vitória é alegre', 'alegria'),
    ('incentivar é humano', 'alegria'),
    ('que ânimo', 'alegria'),
    ('é uma honra para o país', 'alegria'),
    ('o otimismo nos levar a ação', 'alegria'),
    ('passamos ao encorajamento e à lucidez', 'alegria'),
    ('aquele que nunca viu a alegria nunca reconhecerá a tristeza', 'alegria'),
    ('cuidado com a alegria ela é contagiante', 'alegria'),

    ('Estou radiante de felicidade, minha equipe ganhou o campeonato!', 'felicidade'),
    ('Nada me faz mais feliz do que ver o pôr do sol na praia.', 'felicidade'),
    ('Consegui o emprego dos meus sonhos, estou nas nuvens!', 'felicidade'),
    ('Que alegria saber que todos estão seguros e saudáveis.', 'felicidade'),
    ('Hoje é um dia incrível para celebrar a vida e a felicidade.', 'felicidade'),
    ('Estou tão feliz que consegui realizar meu sonho de viajar para o exterior.', 'felicidade'),
    ('Recebi uma notícia maravilhosa que encheu meu coração de alegria.', 'felicidade'),
    ('Sinto uma felicidade imensa ao estar rodeado de amigos e familiares.', 'felicidade'),
    ('Cada momento ao lado da pessoa amada é uma fonte inesgotável de felicidade.', 'felicidade'),
    ('Ganhei um prêmio inesperado e estou pulando de felicidade!', 'felicidade'),
    ('A felicidade transborda quando vejo o sorriso no rosto das crianças.', 'felicidade'),
    ('Hoje é um daqueles dias em que tudo dá certo, estou extremamente feliz!', 'felicidade'),
    ('Realizei um feito que parecia impossível, estou nas nuvens de felicidade.', 'felicidade'),
    ('A sensação de conquistar meus objetivos é simplesmente incrível.', 'felicidade'),
    ('Mal posso conter a felicidade ao receber tantos elogios pelo meu trabalho.', 'felicidade'),
    ('Estou tão feliz que até as estrelas parecem brilhar mais intensamente.', 'felicidade'),
    ('A alegria de estar cercado por pessoas que amo é indescritível.', 'felicidade'),
    ('Hoje é um daqueles dias que vou guardar na memória para sempre.', 'felicidade'),
    ('Agradeço por cada momento de felicidade que a vida me proporciona.', 'felicidade'),
    ('O simples fato de estar vivo já é motivo suficiente para ser feliz.', 'felicidade'),
    ('Sinto uma imensa gratidão por todas as bênçãos que a vida me oferece.', 'felicidade'),
    ('A felicidade está nas pequenas coisas, como um abraço apertado.', 'felicidade'),
    ('A realização de um sonho é como um raio de sol iluminando meu coração.', 'felicidade'),
    ('A felicidade é contagiosa, e hoje ela se espalhou por todos os lados.', 'felicidade'),
    ('Saber que faço a diferença na vida de alguém me enche de alegria.', 'felicidade'),
    ('Viver o presente com gratidão é a chave para uma vida plena de felicidade.', 'felicidade'),
    ('A felicidade não tem preço, e hoje estou rico de momentos especiais.', 'felicidade'),
    ('Cada dia é uma nova oportunidade de ser feliz, e eu aproveito ao máximo.', 'felicidade'),
    ('Estou tão feliz que meu coração parece querer pular para fora do peito.', 'felicidade'),
    ('A felicidade é um estado de espírito, e o meu está radiante hoje.', 'felicidade'),
    ('A vida é uma jornada repleta de momentos felizes, e eu celebro cada um deles.', 'felicidade'),
    ('A felicidade está em abraçar o presente.', 'felicidade'),
    ('A alegria de uma xícara de café quente pela manhã.', 'felicidade'),
    ('A felicidade está em compartilhar risadas com amigos.', 'felicidade'),
    ('A alegria de uma criança brincando despreocupada.', 'felicidade'),
    ('A felicidade está em encontrar beleza na simplicidade.', 'felicidade'),
    ('A alegria de um pôr do sol deslumbrante.', 'felicidade'),
    ('A felicidade está em abraçar a gratidão.', 'felicidade'),
    ('A alegria de ver um sorriso genuíno.', 'felicidade'),
    ('A felicidade está em desfrutar de uma refeição deliciosa.', 'felicidade'),
    ('A alegria de receber elogios sinceros.', 'felicidade'),
    ('A felicidade está em fazer o que você ama.', 'felicidade'),
    ('A alegria de realizar um sonho de infância.', 'felicidade'),
    ('A felicidade está em viver no momento presente.', 'felicidade'),
    ('A alegria de estar rodeado de pessoas que você ama.', 'felicidade'),
    ('A felicidade está em compartilhar momentos especiais.', 'felicidade'),
    ('A alegria de se apaixonar.', 'felicidade'),
    ('A felicidade está em encontrar significado na vida.', 'felicidade'),
    ('A alegria de dançar como ninguém está assistindo.', 'felicidade'),
    ('A felicidade está em abraçar a liberdade de ser você mesmo.', 'felicidade'),
    ('A alegria de superar desafios e alcançar objetivos.', 'felicidade'),
    ('A felicidade está em ser grato pelo que você tem.', 'felicidade'),
    ('A alegria de passar tempo ao ar livre em um dia ensolarado.', 'felicidade'),
    ('A felicidade está em abraçar a jornada da vida.', 'felicidade'),
    ('A alegria de fazer uma boa ação.', 'felicidade'),
    ('A felicidade está em abraçar a simplicidade da vida.', 'felicidade'),
    ('A alegria de ver o florescer da primavera.', 'felicidade'),
    ('A felicidade está em celebrar conquistas.', 'felicidade'),
    ('A alegria de viajar e explorar o mundo.', 'felicidade'),
    ('A felicidade está em fazer alguém sorrir.', 'felicidade'),
    ('A alegria de se perder em um bom livro.', 'felicidade'),
    ('A felicidade está em encontrar a paz interior.', 'felicidade'),
    ('A alegria de rir até doer a barriga.', 'felicidade'),
    ('A felicidade está em viver com um coração grato.', 'felicidade'),
    ('A alegria de ouvir sua música favorita.', 'felicidade'),
    ('A felicidade está em abraçar a diversão.', 'felicidade'),
    ('A alegria de receber uma surpresa inesperada.', 'felicidade')}

baseteste = [
    ('Seu sorriso é a coisa mais linda que já vi.', 'amor'),
    ('Cada momento ao seu lado é como um sonho realizado.', 'amor'),
    ('Meu coração transborda de amor por você.', 'amor'),
    ('Sua presença ilumina meus dias e aquece meu coração.', 'amor'),
    ('O amor que sinto por você é mais forte que qualquer adversidade.', 'amor'),
    ('Seu abraço é o meu lugar seguro, onde encontro paz e felicidade.', 'amor'),
    ('Nunca imaginei que pudesse amar alguém tão intensamente.', 'amor'),
    ('Você é a pessoa que completa o meu mundo e faz tudo ter sentido.', 'amor'),
    ('Cada pequeno gesto seu é um lembrete do quanto eu te amo.', 'amor'),
    ('A cada dia que passa, meu amor por você só cresce.', 'amor'),
    ('Você é a razão do meu sorriso e a fonte da minha felicidade.', 'amor'),
    ('Ao seu lado, cada momento se torna especial e inesquecível.', 'amor'),
    ('Nossa história de amor é a mais linda que já vivi.', 'amor'),
    ('Seu amor é o combustível que me impulsiona a ser uma pessoa melhor.', 'amor'),
    ('A simples ideia de te perder me parte o coração.', 'amor'),
    ('Você é meu porto seguro, meu amor eterno.', 'amor'),
    ('Cada dia ao seu lado é uma bênção que agradeço todos os dias.', 'amor'),
    ('Meu coração pertence a você e a mais ninguém.', 'amor'),
    ('A maneira como você me olha faz meu coração acelerar.', 'amor'),
    ('Com você, descobri o verdadeiro significado do amor.', 'amor'),
    ('Nossas almas se encontraram e se apaixonaram profundamente.', 'amor'),
    ('Sua presença é a luz que ilumina os dias mais escuros.', 'amor'),
    ('Agradeço todos os dias por ter você ao meu lado.', 'amor'),
    ('Você é a pessoa dos meus sonhos, meu amor eterno.', 'amor'),
    ('Cada segundo contigo é um presente que guardo no coração.', 'amor'),
    ('Meu amor por você é eterno, como o céu infinito.', 'amor'),
    ('A vida ao seu lado é uma jornada repleta de amor e felicidade.', 'amor'),
    ('O amor que sinto por você é a força que move minha vida.', 'amor'),
    ('Nada se compara à felicidade que você me proporciona.', 'amor'),
    ('Seu amor é a música que embala a trilha sonora da minha vida.', 'amor'),
    ('Ao seu lado, descobri o verdadeiro significado da plenitude.', 'amor'),
    ('Cada detalhe seu é motivo para eu me apaixonar ainda mais.', 'amor'),
    ('Você é o sonho que nunca soube que tinha, meu amor.', 'amor'),
    ('Amar você é a melhor escolha que já fiz na vida.', 'amor'),

    ('Estou radiante de alegria com essa notícia maravilhosa!', 'alegria'),
    ('Cada conquista me enche de alegria e gratidão.', 'alegria'),
    ('A felicidade transborda quando estou cercado por quem amo.', 'alegria'),
    ('Esse momento de vitória é a razão da minha alegria.', 'alegria'),
    ('Estou tão feliz que parece que vou voar de tanta alegria!', 'alegria'),
    ('Nada supera a alegria de estar ao lado das pessoas que amo.', 'alegria'),
    ('A alegria de viver intensamente é o segredo da felicidade.', 'alegria'),
    ('Recebi uma surpresa que encheu meu coração de alegria.', 'alegria'),
    ('Meu coração transborda de alegria e gratidão por este momento.', 'alegria'),
    ('A alegria de ver um sonho realizado é indescritível.', 'alegria'),
    ('A alegria de compartilhar conquistas é a mais pura felicidade.', 'alegria'),
    ('Cada pequena vitória é motivo para celebrar e espalhar alegria.', 'alegria'),
    ('A alegria de estar com quem amamos é o maior presente da vida.', 'alegria'),
    ('A alegria é a trilha sonora da minha vida.', 'alegria'),
    ('Estou tão feliz que meu coração parece querer pular de alegria.', 'alegria'),
    ('A alegria de viver o presente é o segredo para uma vida plena.', 'alegria'),
    ('A alegria de encontrar amigos verdadeiros é incomparável.', 'alegria'),
    ('Estou vivendo um momento de alegria intensa e contagiante.', 'alegria'),
    ('Cada dia ao seu lado é uma festa de alegria e amor.', 'alegria'),
    ('A alegria de realizar sonhos é a melhor sensação do mundo.', 'alegria'),
    ('A alegria está nas pequenas coisas, como um abraço apertado.', 'alegria'),
    ('A alegria de superar desafios é o que torna a vida emocionante.', 'alegria'),
    ('Estou radiante de alegria por alcançar meus objetivos.', 'alegria'),
    ('A alegria de ser quem sou é o que me impulsiona todos os dias.', 'alegria'),
    ('A alegria de fazer o bem é o que dá significado à minha vida.', 'alegria'),
    ('Estou transbordando de alegria com as boas notícias que recebi.', 'alegria'),
    ('A alegria de ter amigos leais é um tesouro incomparável.', 'alegria'),
    ('A alegria de amar e ser amado é a essência da vida.', 'alegria'),
    ('A cada sorriso, sinto a alegria iluminar meu coração.', 'alegria'),
    ('A alegria de estar cercado por pessoas positivas é contagiante.', 'alegria'),
    ('A alegria de aprender e crescer faz parte da jornada da vida.', 'alegria'),

    ('Que felicidade encontrar você aqui!', 'felicidade'),
    ('Estou radiante de felicidade com essa notícia incrível!', 'felicidade'),
    ('A alegria que sinto é indescritível!', 'felicidade'),
    ('Estou transbordando de felicidade com essa realização.', 'felicidade'),
    ('Cada momento ao seu lado é um pedaço de felicidade.', 'felicidade'),
    ('A felicidade de alcançar meus objetivos é imensa.', 'felicidade'),
    ('A felicidade está estampada no meu rosto hoje.', 'felicidade'),
    ('Sua presença traz uma onda de felicidade para minha vida.', 'felicidade'),
    ('Que alegria saber que todos estão bem e saudáveis.', 'felicidade'),
    ('A felicidade de compartilhar boas notícias é contagiante.', 'felicidade'),
    ('Estou tão feliz que meu coração parece querer pular de alegria!', 'felicidade'),
    ('A alegria de viver o presente é o segredo para uma vida plena.', 'felicidade'),
    ('A alegria de realizar sonhos é a melhor sensação do mundo.', 'felicidade'),
    ('A alegria está nas pequenas coisas, como um abraço apertado.', 'felicidade'),
    ('A felicidade de superar desafios é o que torna a vida emocionante.', 'felicidade'),
    ('Estou radiante de felicidade por alcançar meus objetivos.', 'felicidade'),
    ('A felicidade de ser quem sou é o que me impulsiona todos os dias.', 'felicidade'),
    ('A felicidade de fazer o bem é o que dá significado à minha vida.', 'felicidade'),
    ('Estou transbordando de felicidade com as boas notícias que recebi.', 'felicidade'),
    ('A felicidade de ter amigos leais é um tesouro incomparável.', 'felicidade'),
    ('A felicidade de amar e ser amado é a essência da vida.', 'felicidade'),
    ('A cada sorriso, sinto a felicidade iluminar meu coração.', 'felicidade'),
    ('A felicidade de estar cercado por pessoas positivas é contagiante.', 'felicidade'),
    ('A felicidade de aprender e crescer faz parte da jornada da vida.', 'felicidade')]

base = [
    ('eu sou admirada por muitos', 'alegria'),
    ('me sinto completamente amado', 'amor'),
    ('amar é maravilhoso', 'amor'),
    ('estou me sentindo muito animado novamente', 'alegria'),
    ('eu estou muito bem hoje', 'alegria'),
    ('que belo dia para dirigir um carro novo', 'alegria'),
    ('o dia está muito bonito', 'alegria'),
    ('estou contente com o resultado do teste que fiz ontem', 'alegria'),
    ('o amor é lindo', 'amor'),
    ('nossa amizade e amor vai durar para sempre', 'amor'),
    ('estou amedrontado', 'medo'),
    ('ele está me ameaçando há dias', 'medo'),
    ('isso me deixa apavorada', 'medo'),
    ('este lugar é apavorante', 'medo'),
    ('se perdermos outro jogo seremos eliminados e isso me deixa com pavor', 'medo'),
    ('tome cuidado com o lobisomem', 'medo'),
    ('se eles descobrirem estamos encrencados', 'medo'),
    ('estou tremendo de medo', 'medo'),
    ('eu tenho muito medo dele', 'medo'),
    ('estou com medo do resultado dos meus testes', 'medo')
]

stopwordsnltk = nltk.corpus.stopwords.words('portuguese')
stopwordsnltk.append('vou')
stopwordsnltk.append('tão')


def aplicastemmer(texto):
    stemmer = nltk.stem.RSLPStemmer()
    frasessstemming = []
    for (palavras, emocao) in texto:
        comstemming = [str(stemmer.stem(p)) for p in palavras.split() if p not in stopwordsnltk]
        frasessstemming.append((comstemming, emocao))
    return frasessstemming


frasescomstemmingtreinamento = aplicastemmer(basetreinamento)
frasescomstemmingteste = aplicastemmer(baseteste)


def buscapalavras(frases):
    todaspalavras = []
    for (palavras, emocao) in frases:
        todaspalavras.extend(palavras)
    return todaspalavras


palavras = buscapalavras(frasescomstemmingteste)


def buscafrequencia(palavras):
    palavras = nltk.FreqDist(palavras)
    return palavras


frequencia = buscafrequencia(palavras)


def buscapalavrasunicas(frequencia):
    freq = frequencia.keys()
    return freq


palavrasunicas = buscapalavrasunicas(frequencia)


def extratorpalavras(documento):
    doc = set(documento)
    caracteristicas = {}
    for palavras in palavrasunicas:
        caracteristicas['%s' % palavras] = (palavras in doc)
    return caracteristicas


basecompletatreinamento = nltk.classify.apply_features(extratorpalavras, frasescomstemmingtreinamento)
basecompletateste = nltk.classify.apply_features(extratorpalavras, frasescomstemmingteste)

classificador = nltk.NaiveBayesClassifier.train(basecompletatreinamento)

# Cabeçalho
st.subheader('Análise de Sentimento')

# Texto do usuário
texto_input = st.text_input('Insira o texto para análise (Aperte enter para enviar): ')

# Criando um objeto RSLPStemmer
stemmer = nltk.stem.RSLPStemmer()

# Processamento do texto do usuário
texto_processado = [str(stemmer.stem(p)) for p in texto_input.split() if p not in stopwordsnltk]

# Extrai as características do texto do usuário
caracteristicas_usuario = extratorpalavras(texto_processado)

sentimento = classificador.classify(caracteristicas_usuario)

# Imprime o resultado
st.write("Sentimento do texto:", sentimento)

# # Cabeçalho
# st.subheader('Informações dos dados')
#
# # Texto do usuário
# texto_input = st.sidebar.text_input("Insira o texto para análise: ")
# texto = texto_input
#
# # Loop
# # for texto in textos:
# sentimento = (texto).get_text().strip()
#
# analise_sentimento =
