package ude.master.thesis.stance_detection.wordembeddings;

import java.io.File;
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

import ude.master.thesis.stance_detection.util.ProjectPaths;
import ude.master.thesis.stance_detection.util.StanceDetectionDataReader;

public class DocToVec2WithArguments {

	final static Logger logger = Logger.getLogger(DocToVec2WithArguments.class);

	private static ParagraphVectors paragraphVectors;

	private static List<String> paragraphsList;

	private static List<String> labelsList;

	private static Map<String, String> titleIdMap;

	private TokenizerFactory tokenizerFactory;
	private ParagraphVectors vec;
	
	public DocToVec2WithArguments() throws IOException{
		paragraphsList = new ArrayList<>();
		labelsList = new ArrayList<>();
		titleIdMap = new HashMap<>();

		extractParagraphLabels(paragraphsList, labelsList, titleIdMap);
		paragraphVectors = loadParagraphVectors();
	}

	public ParagraphVectors buildParagraphVectors(List<String> paragraphsList, List<String> labelSourceList) {
		System.out.println("Here");
		SentenceIterator iter = new CollectionSentenceIterator(paragraphsList);
		AbstractCache<VocabWord> cache = new AbstractCache<VocabWord>();

		tokenizerFactory = new DefaultTokenizerFactory();
		tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());

		LabelsSource source = new LabelsSource(labelSourceList);

		vec = new ParagraphVectors.Builder().minWordFrequency(2).iterations(10).epochs(10).layerSize(200)
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
		System.out.println(vec);
		return vec;
	}

	public static void main(String[] args) throws IOException {

		// ================================================================
		paragraphsList = new ArrayList<>();
		labelsList = new ArrayList<>();
		titleIdMap = new HashMap<>();

		extractParagraphLabels(paragraphsList, labelsList, titleIdMap);
		// ===============================================================================

		/** ==== Test the word embeddings ==== **/
		// bulding the paragraph vectors (only once and then saving them)
		DocToVec2WithArguments paraVec = new DocToVec2WithArguments();

		ParagraphVectors docVec = paraVec.buildParagraphVectors(paragraphsList, labelsList);

		WordVectorSerializer.writeParagraphVectors(docVec, "resources/arg_docvec200_081017");

		// Loading the saved paragraph vectors
		paragraphVectors = loadParagraphVectors();

		System.out.println(paragraphVectors.similarWordsInVocabTo("merekel", 0.9));
		System.out.println(paragraphVectors.similarWordsInVocabTo("syria", 0.9));
		System.out.println(paragraphVectors.similarWordsInVocabTo("iphone", 0.8));
	}

	public static ParagraphVectors loadParagraphVectors() throws IOException {
		return WordVectorSerializer.readParagraphVectors("resources/docvec200_160917");
	}

	/**
	 * 
	 * @param paragraphsList
	 * @param labelsList
	 * @throws IOException
	 */
	public static void extractParagraphLabels(List<String> paragraphsList, List<String> labelsList,
			Map<String, String> titleIdMap) throws IOException {
		StanceDetectionDataReader sddr = new StanceDetectionDataReader(true, true,
				ProjectPaths.TRAIN_STANCES_PREPROCESSED, ProjectPaths.ARGUMENTED_BODIES_TRAIN,
				ProjectPaths.TEST_STANCESS_PREPROCESSED, ProjectPaths.ARGUMENTED_BODIES_TEST);

		// paragraphsList hold all the bodies and all the titles as complete
		// strings
		// labelList holds all the labels for title and bodies
		// Training Data
		Map<Integer, String> trainingIdBoyMap = sddr.getTrainIdBodyMap();
		List<List<String>> stancesTraining = sddr.getTrainStances();

		// Test data
		HashMap<Integer, String> testIdBoyMap = sddr.getTestIdBodyMap();
		List<List<String>> stanceTest = sddr.getTestStances();

		Map<Integer, String> allBodies = new HashMap<>();
		for (Entry<Integer, String> e : trainingIdBoyMap.entrySet()) {
			allBodies.put(e.getKey(), e.getValue());
		}
		for (Entry<Integer, String> e : testIdBoyMap.entrySet()) {
			allBodies.put(e.getKey(), e.getValue());
		}

		List<List<String>> allStances = stancesTraining;
		allStances.addAll(stanceTest);

		// adding the bodies an and their labels
		// using the same ids for the bodies
		for (Entry<Integer, String> e : allBodies.entrySet()) {
			paragraphsList.add(e.getValue());
			labelsList.add(String.valueOf(e.getKey()));
		}

		int i = 0;
		int j = 0; // stances

		List<List<String>> titleBodyPairs = new ArrayList<>();

		// adding the title and their labels
		// using title_i as a label for titles
		for (List<String> stance : allStances) {
			// keys are the titles and they are repeated in the set
			// So if the title don't have a label yet
			if (!titleIdMap.containsKey(stance.get(0))) {
				titleIdMap.put(stance.get(0), "title_" + i); // then give it a
																// label
				paragraphsList.add(stance.get(0));
				labelsList.add("title_" + i); // and add it to the labels list

				// and then add it to the list of title labels and body labels
				// list
				List<String> pair = new ArrayList<>();
				pair.add("title_" + i);
				pair.add(stance.get(1));
				titleBodyPairs.add(pair);
				i++;

			} else { // the title already has a label

				List<String> pair = new ArrayList<>();
				pair.add(titleIdMap.get(stance.get(0)));// add the title label
														// (after getting it
														// from the labels map)
				pair.add(stance.get(1)); // add the body_id
				titleBodyPairs.add(pair); // add the pair
			}
			j++;
			if (j == 49972)
				System.out.println(i);
		}

	}

	public double[] getTitleParagraphVecByLabel(String titleTxt) {
		INDArray titleVec = paragraphVectors.getLookupTable().vector(titleIdMap.get(titleTxt));
		return titleVec.data().asDouble();
	}

	public double[] getBodyParagraphVecByLabel(String bodyId) {
		INDArray bodyVec = paragraphVectors.getLookupTable().vector(bodyId);
		return bodyVec.data().asDouble();
	}

}
