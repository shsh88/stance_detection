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
import ude.master.thesis.stance_detection.util.BodySummerizer2;
import ude.master.thesis.stance_detection.util.ProjectPaths;
import ude.master.thesis.stance_detection.util.StanceDetectionDataReader;

public class RelatedUnrelatedFeatureGenerator2 {

	private static Map<Integer, String> trainIdBodyMap = new HashMap<Integer, String>();
	private static List<List<String>> trainingStances = new ArrayList<>();
	private static HashMap<Integer, String> testIdBodyMap = new HashMap<>();
	private static List<List<String>> testStances = new ArrayList<List<String>>();
	private static HashMap<Integer, Map<Integer, String>> trainingSummIdBoyMap = new HashMap<>();
	private static HashMap<Integer, Map<Integer, String>> testSummIdBoyMap = new HashMap<>();

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
	public void getWordOverlapFeatureVector(List<List<String>> stances,
			HashMap<Integer, Map<Integer, String>> summIdBoyMap, String filepath) throws Exception {
		// HashMap to save values
		FileHashMap<String, ArrayList<Double>> wordOverlaps = new FileHashMap<String, ArrayList<Double>>(filepath,
				FileHashMap.FORCE_OVERWRITE);

		List<String[]> entries = new ArrayList<>();
		List<String> csvHeader = new ArrayList<>();
		csvHeader.add("title");
		csvHeader.add("Body ID");
		csvHeader.add("Stance");

		for (int i = 0; i < 9; i++) {
			csvHeader.add("w_overlap_" + i);
		}

		entries.add(csvHeader.toArray(new String[0]));

		for (List<String> s : stances) {
			List<String> entry = new ArrayList<>();
			String title = s.get(0);
			entry.add(title);
			entry.add(s.get(1));
			entry.add(s.get(2));

			Map<Integer, String> bodyParts = summIdBoyMap.get(Integer.valueOf(s.get(1)));

			ArrayList<Double> featureVec = new ArrayList<>();
			for (int i = 1; i <= 3; i++) {
				if ((i == 1) || (i == 3)) {
					String part = bodyParts.get(i);
					List<Double> subVec = getOverlapfeaturePart(title, part, i);
					featureVec.addAll(subVec);

					for (Double v : subVec)
						entry.add(v.toString());
				} else {
					Double fAll = getOverlapfeatureAll(title, bodyParts.get(i));
					featureVec.add(fAll);

					entry.add(fAll.toString());
				}
			}
			if (entry.size() != 12)
				throw new Exception("not 9 features");

			wordOverlaps.put(title + s.get(1), featureVec);

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

	private List<Double> getOverlapfeaturePart(String title, String part, int partNo) {
		List<Double> subVec = new ArrayList<>();

		if (part.equals(""))
			if (partNo == 1) {
				Double[] vec = new Double[BodySummerizer2.NUM_SENT_BEG];
				Arrays.fill(vec, -100.0);
				return Arrays.asList(vec);
			} else if (partNo == 3) {
				Double[] vec = new Double[BodySummerizer2.NUM_SENT_END];
				Arrays.fill(vec, -100.0);
				return Arrays.asList(vec);
			}

		if ((partNo == 1) || (partNo == 3)) {
			if (pipeline == null)
				pipeline = getStanfordPipeline();

			Annotation doc = new Annotation(part);
			pipeline.annotate(doc);
			List<CoreMap> sentences = doc.get(SentencesAnnotation.class);

			for (CoreMap s : sentences) {
				double f = FeatureExtractorWithModifiedBL.getWordOverlapFeature(title, s.toString());
				subVec.add(f);
			}

			if (partNo == 1)
				while (subVec.size() < BodySummerizer2.NUM_SENT_BEG)
					subVec.add(-100.0);

			if (partNo == 3)
				while (subVec.size() < BodySummerizer2.NUM_SENT_END)
					subVec.add(-100.0);
		}

		return subVec;
	}

	private static StanfordCoreNLP getStanfordPipeline() {
		Properties props = new Properties();
		props.setProperty("annotators", "tokenize,ssplit,pos,lemma");
		StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
		return pipeline;
	}

	public void getBinCoOccurFeatureVector(List<List<String>> stances,
			HashMap<Integer, Map<Integer, String>> summIdBoyMap, String filepath) throws Exception {
		// HashMap to save values
		FileHashMap<String, ArrayList<Integer>> wordCoOcc = new FileHashMap<String, ArrayList<Integer>>(filepath,
				FileHashMap.FORCE_OVERWRITE);

		List<String[]> entries = new ArrayList<>();
		List<String> csvHeader = new ArrayList<>();
		csvHeader.add("title");
		csvHeader.add("Body ID");
		csvHeader.add("Stance");

		for (int i = 0; i < 9; i++) {
			csvHeader.add("bin_co_stop_" + i);
		}

		entries.add(csvHeader.toArray(new String[0]));

		for (List<String> s : stances) {
			List<String> entry = new ArrayList<>();
			String title = s.get(0);
			entry.add(title);
			entry.add(s.get(1));
			entry.add(s.get(2));

			Map<Integer, String> bodyParts = summIdBoyMap.get(Integer.valueOf(s.get(1)));

			ArrayList<Integer> featureVec = new ArrayList<>();
			for (int i = 1; i <= 3; i++) {
				if ((i == 1) || (i == 3)) {
					String part = bodyParts.get(i);
					List<Integer> subVec = getCoOccfeaturePart(title, part, i);
					featureVec.addAll(subVec);

					for (Integer v : subVec)
						entry.add(v.toString());
				} else {
					List<Integer> fAll = getCoOccfeatureAll(title, bodyParts.get(i));
					featureVec.addAll(fAll);

					entry.add(fAll.get(0).toString());
					entry.add(fAll.get(1).toString());
				}
			}
	
			if (entry.size() != 13)
				throw new Exception("not 10 features");

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

	private List<Integer> getCoOccfeaturePart(String title, String part, int partNo) {
		List<Integer> subVec = new ArrayList<>();

		if (part.equals("")){
			System.out.println("That cannot happen!");
			if (partNo == 1) {
				Integer[] vec = new Integer[BodySummerizer2.NUM_SENT_BEG];
				Arrays.fill(vec, -100);
				return Arrays.asList(vec);
			} else if (partNo == 3) {
				Integer[] vec = new Integer[BodySummerizer2.NUM_SENT_END];
				Arrays.fill(vec, -100);
				return Arrays.asList(vec);
			}
		}

		if ((partNo == 1) || (partNo == 3)) {
			if (pipeline == null)
				pipeline = getStanfordPipeline();

			Annotation doc = new Annotation(part);
			pipeline.annotate(doc);
			List<CoreMap> sentences = doc.get(SentencesAnnotation.class);

			for (CoreMap s : sentences) {
				int f = FeatureExtractorWithModifiedBL.getSentenceBinaryCoOccurenceStopFeatures(title, s.toString());
				subVec.add(f);
			}

			if (partNo == 1)
				while (subVec.size() < BodySummerizer2.NUM_SENT_BEG)
					subVec.add(-100);

			if (partNo == 3)
				while (subVec.size() < BodySummerizer2.NUM_SENT_END)
					subVec.add(-100);
		}

		return subVec;
	}

	public void getCharGramFeatureVector(List<List<String>> stances,
			HashMap<Integer, Map<Integer, String>> summIdBoyMap, String filepath) throws Exception {
		// HashMap to save values
		FileHashMap<String, ArrayList<Integer>> charGrams = new FileHashMap<String, ArrayList<Integer>>(filepath,
				FileHashMap.FORCE_OVERWRITE);

		List<String[]> entries = new ArrayList<>();
		List<String> csvHeader = new ArrayList<>();
		csvHeader.add("title");
		csvHeader.add("Body ID");
		csvHeader.add("Stance");

		int featSize = (3 * BodySummerizer2.NUM_SENT_BEG) + (3 * BodySummerizer2.NUM_SENT_END)
				+ 3 * 3;
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

			Map<Integer, String> bodyParts = summIdBoyMap.get(Integer.valueOf(s.get(1)));

			ArrayList<Integer> featureVec = new ArrayList<>();
			for (int i = 1; i <= 3; i++) {
				if ((i == 1) || (i == 3)) {
					String part = bodyParts.get(i);
					List<Integer> subVec = getCharGramFeaturePart(title, part, i);
					featureVec.addAll(subVec);

					for (Integer v : subVec)
						entry.add(v.toString());
				} else {
					List<Integer> fAll = getCharGramFeatureAll(title, bodyParts.get(i));
					featureVec.addAll(fAll);

					for (Integer ff : fAll)
						entry.add(ff.toString());
				}
			}
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

	private List<Integer> getCharGramFeaturePart(String title, String part, int partNo) {
		int[] cgramSizes = { 4, 8, 16 };
		List<Integer> subVec = new ArrayList<>();

		if (part.equals("")){
			System.out.println(("This cannot happen!"));
			if (partNo == 1) {
				Integer[] vec = new Integer[BodySummerizer2.NUM_SENT_BEG * cgramSizes.length];
				Arrays.fill(vec, -100);
				return Arrays.asList(vec);
			} else if (partNo == 3) {
				Integer[] vec = new Integer[BodySummerizer2.NUM_SENT_END * cgramSizes.length];
				Arrays.fill(vec, -100);
				return Arrays.asList(vec);
			}
		}

		if ((partNo == 1) || (partNo == 3)) {
			if (pipeline == null)
				pipeline = getStanfordPipeline();

			Annotation doc = new Annotation(part);
			pipeline.annotate(doc);
			List<CoreMap> sentences = doc.get(SentencesAnnotation.class);

			for (CoreMap s : sentences) {
				List<Integer> features = new ArrayList<>();
				for (int size : cgramSizes) {
					int f = FeatureExtractorWithModifiedBL.getSentenceCharGramsFeatures(title, s.toString(), size);
					features.add(f);
				}
				subVec.addAll(features);
			}

			if (partNo == 1)
				while (subVec.size() < (BodySummerizer2.NUM_SENT_BEG * cgramSizes.length))
					subVec.add(-100);

			if (partNo == 3)
				while (subVec.size() < (BodySummerizer2.NUM_SENT_END * cgramSizes.length))
					subVec.add(-100);
		}

		return subVec;
	}

	public void getNGramFeatureVector(List<List<String>> stances, HashMap<Integer, Map<Integer, String>> summIdBoyMap,
			String filepath) throws Exception {
		// HashMap to save values
		FileHashMap<String, ArrayList<Integer>> nGrams = new FileHashMap<String, ArrayList<Integer>>(filepath,
				FileHashMap.FORCE_OVERWRITE);

		List<String[]> entries = new ArrayList<>();
		List<String> csvHeader = new ArrayList<>();
		csvHeader.add("title");
		csvHeader.add("Body ID");
		csvHeader.add("Stance");

		int featSize = (4 * BodySummerizer2.NUM_SENT_BEG) + (4 * BodySummerizer2.NUM_SENT_END)
				+ 4 * 2;
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

			Map<Integer, String> bodyParts = summIdBoyMap.get(Integer.valueOf(s.get(1)));

			ArrayList<Integer> featureVec = new ArrayList<>();
			for (int i = 1; i <= 3; i++) {
				if ((i == 1) || (i == 3)) {
					String part = bodyParts.get(i);
					List<Integer> subVec = getNGramFeaturePart(title, part, i);
					featureVec.addAll(subVec);

					for (Integer v : subVec)
						entry.add(v.toString());
				} else {
					List<Integer> fAll = getNGramFeatureAll(title, bodyParts.get(i));
					featureVec.addAll(fAll);

					for (Integer ff : fAll)
						entry.add(ff.toString());
				}
			}
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

		if (part.equals("")){
			System.out.println("This cannot happen !");
			if (partNo == 1) {
				Integer[] vec = new Integer[BodySummerizer2.NUM_SENT_BEG * ngramSizes.length];
				Arrays.fill(vec, -100);
				return Arrays.asList(vec);
			} else if (partNo == 3) {
				Integer[] vec = new Integer[BodySummerizer2.NUM_SENT_END * ngramSizes.length];
				Arrays.fill(vec, -100);
				return Arrays.asList(vec);
			}
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
				while (subVec.size() < (BodySummerizer2.NUM_SENT_BEG * ngramSizes.length))
					subVec.add(-100);

			if (partNo == 3)
				while (subVec.size() < (BodySummerizer2.NUM_SENT_END * ngramSizes.length))
					subVec.add(-100);
		}

		return subVec;
	}

	public static void loadData() throws IOException {
		StanceDetectionDataReader sddr = new StanceDetectionDataReader(true, true,
				ProjectPaths.TRAIN_STANCES_PREPROCESSED, ProjectPaths.SUMMARIZED_TRAIN_BODIES2,
				ProjectPaths.TEST_STANCESS_PREPROCESSED, ProjectPaths.SUMMARIZED_TEST_BODIES2);

		trainingSummIdBoyMap = sddr.readSummIdBodiesMap(new File(ProjectPaths.SUMMARIZED_TRAIN_BODIES2));
		testSummIdBoyMap = sddr.readSummIdBodiesMap(new File(ProjectPaths.SUMMARIZED_TEST_BODIES2));

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

	public static FileHashMap<String, ArrayList<Double>> loadWordsOverlapsFeaturesAsHashFile(String hashFileName)
			throws FileNotFoundException, ObjectExistsException, ClassNotFoundException, VersionMismatchException,
			IOException {
		FileHashMap<String, ArrayList<Double>> wordOverlaps = new FileHashMap<String, ArrayList<Double>>(hashFileName,
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

		RelatedUnrelatedFeatureGenerator2 rufg = new RelatedUnrelatedFeatureGenerator2();

		// overlaps features
		//rufg.getWordOverlapFeatureVector(trainingStances, trainingSummIdBoyMap, ProjectPaths.TRAIN_WORD_OVERLAPS_PATH2);
		//rufg.getWordOverlapFeatureVector(testStances, testSummIdBoyMap, ProjectPaths.TEST_WORD_OVERLAPS_PATH2);

		// Binary cooccurances stop
		//rufg.getBinCoOccurFeatureVector(trainingStances, trainingSummIdBoyMap, ProjectPaths.TRAIN_COOCC_PATH2);
		//rufg.getBinCoOccurFeatureVector(testStances, testSummIdBoyMap, ProjectPaths.TEST_COOCC_PATH2);

		// Chargrams
		//rufg.getCharGramFeatureVector(trainingStances, trainingSummIdBoyMap, ProjectPaths.TRAIN_CGRAMS_PATH2);
		//rufg.getCharGramFeatureVector(testStances, testSummIdBoyMap, ProjectPaths.TEST_CGRAMS_PATH2);

		// Chargrams
		rufg.getNGramFeatureVector(trainingStances, trainingSummIdBoyMap, ProjectPaths.TRAIN_NGRAMS_PATH2);
		rufg.getNGramFeatureVector(testStances, testSummIdBoyMap, ProjectPaths.TEST_NGRAMS_PATH2);
	}
}
