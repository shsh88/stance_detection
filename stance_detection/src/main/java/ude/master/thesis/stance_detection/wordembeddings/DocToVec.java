package ude.master.thesis.stance_detection.wordembeddings;


import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import org.apache.log4j.Logger;
import org.deeplearning4j.models.embeddings.learning.impl.sequence.DM;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.AbstractCache;
import org.deeplearning4j.text.documentiterator.LabelsSource;
import org.deeplearning4j.text.sentenceiterator.CollectionSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;

import ude.master.thesis.stance_detection.util.StanceDetectionDataReader;

public class DocToVec {
	
	final static Logger logger = Logger.getLogger(DocToVec.class);
	
	private TokenizerFactory tokenizerFactory;
	private ParagraphVectors vec;

	public ParagraphVectors buildParagraphVectors(List<String> tweetMessagesList, List<String> labelSourceList) {
		SentenceIterator iter = new CollectionSentenceIterator(tweetMessagesList);
		AbstractCache<VocabWord> cache = new AbstractCache<VocabWord>();

		tokenizerFactory = new DefaultTokenizerFactory();
		tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());

		LabelsSource source = new LabelsSource(labelSourceList);

		vec = new ParagraphVectors.Builder().minWordFrequency(2).iterations(10).epochs(10).layerSize(100)
				.learningRate(0.025)
				// .minLearningRate(0.001)
				.labelsSource(source)
				// .stopWords(Files.readAllLines(new
				// File("../stopwords.txt").toPath(), Charset.defaultCharset()
				// ))
				.windowSize(10).iterate(iter).trainWordVectors(true).vocabCache(cache)
				// Wahlweise Distributional-BOW(default) oder Distributional
				// Memory new DM<VocabWord>()
				.sequenceLearningAlgorithm(new DM<VocabWord>()).tokenizerFactory(tokenizerFactory)
				// .sampling(0)
				.build();

		vec.fit();

		return vec;
	}
	
	public static void main(String[] args) throws IOException {
		/*List<String> paragraphsList = new ArrayList<>();
		List<String> labelsList = new ArrayList<>();
		
		StanceDetectionDataReader sddr = new StanceDetectionDataReader(true, true);
		
		//Training Data
		Map<Integer, String> bodiesIdsMapTraining = sddr.getTrainIdBodyMap();
		List<List<String>> stancesTraining = sddr.getTrainStances();
		
		//Test data
		HashMap<Integer, String> bodiesIdsMapTest = sddr.getTestIdBodyMap();
		List<List<String>> stanceTest = sddr.getTestStances();
		
		Map<Integer, String> allBodies = bodiesIdsMapTraining;
		allBodies.putAll(bodiesIdsMapTest);
		
		List<List<String>> allStances = stancesTraining;
		allStances.addAll(stanceTest);
		
		for (Entry<Integer, String> e : allBodies.entrySet()) {
			paragraphsList.add(e.getValue());
			labelsList.add(String.valueOf(e.getKey()));
		}
		
		int i = 0;
		int j = 0; //stances
		
		Map<String, String> titleIdMap = new HashMap<>();
		List<List<String>> titleBodyPairs = new ArrayList<>();
		
		for (List<String> stance : allStances) {
			if (!titleIdMap.containsKey(stance.get(0))) {
				titleIdMap.put(stance.get(0),"title_" + i);
				paragraphsList.add(stance.get(0));
				labelsList.add("title_" + i);

				List<String> pair = new ArrayList<>();
				pair.add("title_" + i);
				pair.add(stance.get(1));
				titleBodyPairs.add(pair);
				i++;
			
			} else {

				List<String> pair = new ArrayList<>();
				pair.add(titleIdMap.get(stance.get(0)));
				pair.add(stance.get(1));
				titleBodyPairs.add(pair);
			}
			j++;
			if(j==49972)
				System.out.println(i);
		}*/
		
		DocToVec paraVec = new DocToVec();
		// ParagraphVectors docVec = paraVec.buildParagraphVectors(paragraphsList,
		 //labelsList);
		 //WordVectorSerializer.writeParagraphVectors(docVec, "resources/docvec_070817");

		ParagraphVectors pvecs = WordVectorSerializer.readParagraphVectors("resources/docvec_070817");
		
		System.out.println(pvecs.similarWordsInVocabTo("isis", 0.9));
		System.out.println(pvecs.similarWordsInVocabTo("Steve", 0.9));
		System.out.println(pvecs.similarWordsInVocabTo("money", 0.8));
	}


}
