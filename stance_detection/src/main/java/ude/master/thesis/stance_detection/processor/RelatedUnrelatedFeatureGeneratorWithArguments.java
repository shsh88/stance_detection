package ude.master.thesis.stance_detection.processor;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;

import org.clapper.util.misc.FileHashMap;
import org.clapper.util.misc.ObjectExistsException;
import org.clapper.util.misc.VersionMismatchException;

import com.opencsv.CSVWriter;

import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.util.CoreMap;
import ude.master.thesis.stance_detection.main.ClassifierTools;
import ude.master.thesis.stance_detection.util.FNCConstants;
import ude.master.thesis.stance_detection.util.ProjectPaths;
import ude.master.thesis.stance_detection.util.StanceDetectionDataReader;
import ude.master.thesis.stance_detection.util.TitleAndBodyTextPreprocess;
import weka.core.Attribute;

public class RelatedUnrelatedFeatureGeneratorWithArguments {

	private static Map<Integer, String> trainIdBodyMap = new HashMap<Integer, String>();
	private static List<List<String>> trainingStances = new ArrayList<>();
	private static HashMap<Integer, String> testIdBodyMap = new HashMap<>();
	private static List<List<String>> testStances = new ArrayList<List<String>>();

	StanfordCoreNLP pipeline;

	/**
	 * generates the feature vector which is of size 11: a feature from each
	 * sentence in the beginnning (6 sentences) and a feature for each sentence
	 * at last (4 sentences) and other one feature for the whole bodey if there
	 * is no sentences as expected the feature value is (0) or maybe (-1) is
	 * used
	 * 
	 * @throws Exception
	 */
	public void getWordOverlapFeatureVector(List<List<String>> stances, Map<Integer, String> trainIdBodyMap,
			String filepath) throws Exception {
		// HashMap to save values
		FileHashMap<String, Double> wordOverlaps = new FileHashMap<String, Double>(filepath,
				FileHashMap.FORCE_OVERWRITE);

		List<String[]> entries = new ArrayList<>();
		List<String> csvHeader = new ArrayList<>();
		csvHeader.add("title");
		csvHeader.add("Body ID");
		csvHeader.add("Stance");

		csvHeader.add("w_overlap");

		entries.add(csvHeader.toArray(new String[0]));

		for (List<String> s : stances) {
			List<String> entry = new ArrayList<>();
			String title = s.get(0);
			entry.add(title);
			entry.add(s.get(1));
			entry.add(s.get(2));

			String body = trainIdBodyMap.get(Integer.valueOf(s.get(1)));

			Double fAll = getOverlapfeatureAll(title, body);

			entry.add(fAll.toString());

			wordOverlaps.put(title + s.get(1), fAll);

			entries.add(entry.toArray(new String[0]));
		}

		wordOverlaps.save();
		wordOverlaps.close();

		CSVWriter writer = new CSVWriter(new FileWriter(filepath + ".csv"));
		writer.writeAll(entries);
		writer.flush();
		writer.close();
		System.out.println("saved saved saved");

	}

	private Double getOverlapfeatureAll(String title, String all) {
		return FeatureExtractorWithModifiedBL.getWordOverlapFeature(title, all);
	}

	private static StanfordCoreNLP getStanfordPipeline() {
		Properties props = new Properties();
		props.setProperty("annotators", "tokenize,ssplit,pos,lemma");
		StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
		return pipeline;
	}

	public void getBinCoOccurFeatureVector(List<List<String>> stances, Map<Integer, String> trainIdBodyMap,
			String filepath) throws Exception {
		// HashMap to save values
		FileHashMap<String, ArrayList<Integer>> wordCoOcc = new FileHashMap<String, ArrayList<Integer>>(filepath,
				FileHashMap.FORCE_OVERWRITE);

		List<String[]> entries = new ArrayList<>();
		List<String> csvHeader = new ArrayList<>();
		csvHeader.add("title");
		csvHeader.add("Body ID");
		csvHeader.add("Stance");

		csvHeader.add("bin_co_stop");

		entries.add(csvHeader.toArray(new String[0]));

		for (List<String> s : stances) {
			List<String> entry = new ArrayList<>();
			String title = s.get(0);
			entry.add(title);
			entry.add(s.get(1));
			entry.add(s.get(2));

			String body = trainIdBodyMap.get(Integer.valueOf(s.get(1)));

			ArrayList<Integer> featureVec = new ArrayList<>();

			List<Integer> fAll = getCoOccfeatureAll(title, body);
			featureVec.addAll(fAll);

			entry.add(fAll.get(0).toString());
			entry.add(fAll.get(1).toString());

			if (entry.size() != 5)
				throw new Exception("not 2 features");

			wordCoOcc.put(title + s.get(1), featureVec);

			entries.add(entry.toArray(new String[0]));
		}

		wordCoOcc.save();
		wordCoOcc.close();

		CSVWriter writer = new CSVWriter(new FileWriter(filepath + ".csv"));
		writer.writeAll(entries);
		writer.flush();
		writer.close();
		System.out.println("saved saved saved");

	}

	private List<Integer> getCoOccfeatureAll(String title, String body) {
		return FeatureExtractorWithModifiedBL.getBinaryCoOccurenceStopFeatures(title, body);
	}

	public void getCharGramFeatureVector(List<List<String>> stances, Map<Integer, String> trainIdBodyMap,
			String filepath) throws Exception {
		// HashMap to save values
		FileHashMap<String, ArrayList<Integer>> charGrams = new FileHashMap<String, ArrayList<Integer>>(filepath,
				FileHashMap.FORCE_OVERWRITE);

		List<String[]> entries = new ArrayList<>();
		List<String> csvHeader = new ArrayList<>();
		csvHeader.add("title");
		csvHeader.add("Body ID");
		csvHeader.add("Stance");

		int featSize = 3 * 3;
		for (int i = 0; i < featSize; i++) {
			csvHeader.add("cgram_" + i);
		}

		entries.add(csvHeader.toArray(new String[0]));

		for (List<String> s : stances) {
			List<String> entry = new ArrayList<>();
			String title = s.get(0);
			entry.add(title);
			entry.add(s.get(1));
			entry.add(s.get(2));

			String body = trainIdBodyMap.get(Integer.valueOf(s.get(1)));

			ArrayList<Integer> featureVec = new ArrayList<>();

			List<Integer> fAll = getCharGramFeatureAll(title, body);
			featureVec.addAll(fAll);

			for (Integer ff : fAll)
				entry.add(ff.toString());
			if (entry.size() != (featSize + 3))
				throw new Exception("not " + featSize + " features");

			charGrams.put(title + s.get(1), featureVec);

			entries.add(entry.toArray(new String[0]));
		}

		charGrams.save();
		charGrams.close();

		CSVWriter writer = new CSVWriter(new FileWriter(filepath + ".csv"));
		writer.writeAll(entries);
		writer.flush();
		writer.close();
		System.out.println("saved saved saved");

	}

	private List<Integer> getCharGramFeatureAll(String title, String body) {
		// int[] cgramSizes = { 2, 4, 8, 16 };
		int[] cgramSizes = { 4, 8, 16 };

		List<Integer> features = new ArrayList<>();
		for (int size : cgramSizes) {
			List<Integer> f = FeatureExtractorWithModifiedBL.getCharGramsFeatures(title, body, size);
			features.addAll(f);
		}
		return features;
	}

	public void getNGramFeatureVector(List<List<String>> stances, Map<Integer, String> trainIdBodyMap, String filepath)
			throws Exception {
		// HashMap to save values
		FileHashMap<String, ArrayList<Integer>> nGrams = new FileHashMap<String, ArrayList<Integer>>(filepath,
				FileHashMap.FORCE_OVERWRITE);

		List<String[]> entries = new ArrayList<>();
		List<String> csvHeader = new ArrayList<>();
		csvHeader.add("title");
		csvHeader.add("Body ID");
		csvHeader.add("Stance");

		int featSize = 4 * 2;
		for (int i = 0; i < featSize; i++) {
			csvHeader.add("ngram_" + i);
		}

		entries.add(csvHeader.toArray(new String[0]));

		for (List<String> s : stances) {
			List<String> entry = new ArrayList<>();
			String title = s.get(0);
			entry.add(title);
			entry.add(s.get(1));
			entry.add(s.get(2));

			String body = trainIdBodyMap.get(Integer.valueOf(s.get(1)));

			ArrayList<Integer> featureVec = new ArrayList<>();

			List<Integer> fAll = getNGramFeatureAll(title, body);
			featureVec.addAll(fAll);

			for (Integer ff : fAll)
				entry.add(ff.toString());

			if (entry.size() != (featSize + 3))
				throw new Exception("not " + featSize + " features");

			nGrams.put(title + s.get(1), featureVec);

			entries.add(entry.toArray(new String[0]));
		}

		nGrams.save();
		nGrams.close();

		CSVWriter writer = new CSVWriter(new FileWriter(filepath + ".csv"));
		writer.writeAll(entries);
		writer.flush();
		writer.close();
		System.out.println("saved saved saved");

	}

	private List<Integer> getNGramFeatureAll(String title, String body) {
		int[] ngramSizes = { 2, 4, 8, 16 };
		List<Integer> features = new ArrayList<>();
		for (int size : ngramSizes) {
			List<Integer> f = FeatureExtractorWithModifiedBL.getNGramsFeatures(title, body, size);
			features.addAll(f);
		}
		return features;
	}

	private List<Integer> getNGramFeaturePart(String title, String part, int partNo) {
		int[] ngramSizes = { 2, 4, 8, 16 };
		List<Integer> subVec = new ArrayList<>();

		if (part.equals(""))
			if (partNo == 1) {
				Integer[] vec = new Integer[TitleAndBodyTextPreprocess.NUM_SENT_BEG * ngramSizes.length];
				Arrays.fill(vec, -1);
				return Arrays.asList(vec);
			} else if (partNo == 3) {
				Integer[] vec = new Integer[TitleAndBodyTextPreprocess.NUM_SENT_END * ngramSizes.length];
				Arrays.fill(vec, -1);
				return Arrays.asList(vec);
			}

		if ((partNo == 1) || (partNo == 3)) {
			if (pipeline == null)
				pipeline = getStanfordPipeline();

			Annotation doc = new Annotation(part);
			pipeline.annotate(doc);
			List<CoreMap> sentences = doc.get(SentencesAnnotation.class);

			for (CoreMap s : sentences) {
				List<Integer> features = new ArrayList<>();
				for (int size : ngramSizes) {
					int f = FeatureExtractorWithModifiedBL.getSentenceNGramsFeatures(title, s.toString(), size);
					features.add(f);
				}
				subVec.addAll(features);
			}

			if (partNo == 1)
				while (subVec.size() < (TitleAndBodyTextPreprocess.NUM_SENT_BEG * ngramSizes.length))
					subVec.add(-1);

			if (partNo == 3)
				while (subVec.size() < (TitleAndBodyTextPreprocess.NUM_SENT_END * ngramSizes.length))
					subVec.add(-1);
		}

		return subVec;
	}

	public static void loadData() throws IOException {
		StanceDetectionDataReader sddr = new StanceDetectionDataReader(true, true,
				ProjectPaths.TRAIN_STANCES_PREPROCESSED, ProjectPaths.ARGUMENTED_BODIES_TRAIN,
				ProjectPaths.TEST_STANCESS_PREPROCESSED, ProjectPaths.ARGUMENTED_BODIES_TEST);

		trainIdBodyMap = sddr.getTrainIdBodyMap();
		testIdBodyMap = sddr.getTestIdBodyMap();

		trainingStances = sddr.getTrainStances();

		testStances = sddr.getTestStances();
	}

	public static FileHashMap<String, ArrayList<Integer>> loadCoOccStopFeaturesAsHashFile(String hashFileName)
			throws FileNotFoundException, ObjectExistsException, ClassNotFoundException, VersionMismatchException,
			IOException {
		FileHashMap<String, ArrayList<Integer>> wordCoOcc = new FileHashMap<String, ArrayList<Integer>>(hashFileName,
				FileHashMap.FORCE_OVERWRITE);
		return wordCoOcc;
	}

	public static FileHashMap<String, Double> loadWordsOverlapsFeaturesAsHashFile(String hashFileName)
			throws FileNotFoundException, ObjectExistsException, ClassNotFoundException, VersionMismatchException,
			IOException {
		FileHashMap<String, Double> wordOverlaps = new FileHashMap<String, Double>(hashFileName,
				FileHashMap.FORCE_OVERWRITE);
		return wordOverlaps;
	}

	public static FileHashMap<String, ArrayList<Integer>> loadCharGramsFeaturesAsHashFile(String hashFileName)
			throws FileNotFoundException, ObjectExistsException, ClassNotFoundException, VersionMismatchException,
			IOException {
		FileHashMap<String, ArrayList<Integer>> charGrams = new FileHashMap<String, ArrayList<Integer>>(hashFileName,
				FileHashMap.FORCE_OVERWRITE);
		return charGrams;
	}

	public static FileHashMap<String, ArrayList<Integer>> loadNGramsFeaturesAsHashFile(String hashFileName)
			throws FileNotFoundException, ObjectExistsException, ClassNotFoundException, VersionMismatchException,
			IOException {
		FileHashMap<String, ArrayList<Integer>> nGrams = new FileHashMap<String, ArrayList<Integer>>(hashFileName,
				FileHashMap.FORCE_OVERWRITE);
		return nGrams;
	}

	public static void main(String[] args) throws Exception {
		loadData();

		RelatedUnrelatedFeatureGeneratorWithArguments rufg = new RelatedUnrelatedFeatureGeneratorWithArguments();

		// overlaps features
		rufg.getWordOverlapFeatureVector(trainingStances, trainIdBodyMap, ProjectPaths.TRAIN_ARG_WORD_OVERLAPS_PATH);
		rufg.getWordOverlapFeatureVector(testStances, testIdBodyMap, ProjectPaths.TEST_ARG_WORD_OVERLAPS_PATH);

		// Binary cooccurances stop
		rufg.getBinCoOccurFeatureVector(trainingStances, trainIdBodyMap, ProjectPaths.TRAIN_ARG_COOCC_PATH);
		rufg.getBinCoOccurFeatureVector(testStances, testIdBodyMap, ProjectPaths.TEST_ARG_COOCC_PATH);

		// Chargrams
		rufg.getCharGramFeatureVector(trainingStances, trainIdBodyMap, ProjectPaths.TRAIN_ARG_CGRAMS_PATH);
		rufg.getCharGramFeatureVector(testStances, testIdBodyMap, ProjectPaths.TEST_ARG_CGRAMS_PATH);

		// Chargrams
		rufg.getNGramFeatureVector(trainingStances, trainIdBodyMap, ProjectPaths.TRAIN_ARG_NGRAMS_PATH);
		rufg.getNGramFeatureVector(testStances, testIdBodyMap, ProjectPaths.TEST_ARG_NGRAMS_PATH);
	}
}
