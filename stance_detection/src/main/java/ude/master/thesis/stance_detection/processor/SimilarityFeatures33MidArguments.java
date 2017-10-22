package ude.master.thesis.stance_detection.processor;

import static org.simmetrics.builders.StringDistanceBuilder.with;

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
import org.simmetrics.StringDistance;
import org.simmetrics.StringMetric;
import org.simmetrics.metrics.CosineSimilarity;
import org.simmetrics.metrics.StringMetrics;
import org.simmetrics.simplifiers.Simplifiers;
import org.simmetrics.tokenizers.Tokenizers;

import com.opencsv.CSVWriter;

import edu.mit.jwi.IDictionary;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.util.CoreMap;
import ude.master.thesis.stance_detection.util.BodySummarizerWithArguments;
import ude.master.thesis.stance_detection.util.LeskGlossOverlaps;
import ude.master.thesis.stance_detection.util.ProjectPaths;
import ude.master.thesis.stance_detection.util.StanceDetectionDataReader;

public class SimilarityFeatures33MidArguments {

	private static StringDistance cosSimMetric;
	private static StringMetric cosSimStringMetric;
	private LeskGlossOverlaps lgo;

	StanfordCoreNLP pipeline;

	private static List<List<String>> trainingStances = new ArrayList<>();
	private static List<List<String>> testStances = new ArrayList<List<String>>();
	private static HashMap<Integer, Map<Integer, String>> trainingSummIdBoyMap = new HashMap<>();
	private static HashMap<Integer, Map<Integer, String>> testSummIdBoyMap = new HashMap<>();

	private void initCosSimilarityMetric() {
		System.out.println("init metric");
		// TreeSet<String> stopSet =
		// initializeStopwords("resources/stopwords.txt");

		// Set<String> commonWords = Sets.newHashSet(stopSet);
		cosSimMetric = with(new CosineSimilarity<String>()).simplify(Simplifiers.toLowerCase())
				.simplify(Simplifiers.removeNonWord()).tokenize(Tokenizers.whitespace()).tokenize(Tokenizers.qGram(5))
				.build();
	}

	private void initCosSimilarityStringMetric() {
		System.out.println("init metric");
		cosSimStringMetric = StringMetrics.cosineSimilarity();
	}

	private void initLesk() throws IOException {
		IDictionary dict = LeskGlossOverlaps.getDictionary();
		lgo = new LeskGlossOverlaps(dict);
		lgo.useStopList(true);
		lgo.useLemmatiser(true);
	}

	public void getCosSimFeatureVector(List<List<String>> stances, HashMap<Integer, Map<Integer, String>> summIdBoyMap,
			String filepath) throws Exception {
		// HashMap to save values
		FileHashMap<String, ArrayList<Double>> cosSims = new FileHashMap<String, ArrayList<Double>>(filepath,
				FileHashMap.FORCE_OVERWRITE);

		if (cosSimStringMetric == null)
			initCosSimilarityStringMetric();

		List<String[]> entries = new ArrayList<>();
		List<String> csvHeader = new ArrayList<>();
		csvHeader.add("title");
		csvHeader.add("Body ID");
		csvHeader.add("Stance");

		for (int i = 0; i < 9; i++) {
			csvHeader.add("cos_sim_" + i);
		}

		entries.add(csvHeader.toArray(new String[0]));

		for (List<String> s : stances) {
			List<String> entry = new ArrayList<>();
			String title = s.get(0);
			entry.add(title);
			entry.add(s.get(1));
			entry.add(s.get(2));
			if (summIdBoyMap.containsKey(Integer.valueOf(s.get(1)))) {
				Map<Integer, String> bodyParts = summIdBoyMap.get(Integer.valueOf(s.get(1)));

				ArrayList<Double> featureVec = new ArrayList<>();
				for (int i = 1; i <= 3; i++) {
					if ((i == 1) || (i == 3)) {
						String part = bodyParts.get(i);
						if (part.equals(""))
							System.out.println(s.get(0) + " " + s.get(1));
						List<Double> subVec = getCosSimfeaturePart(title, part, i);
						featureVec.addAll(subVec);

						for (Double v : subVec)
							entry.add(v.toString());

						if (i == 1) {
							Double v = getCosSimfeatureAll(title, part);
							featureVec.add(v);
							entry.add(v.toString());
						}
					} else {
						Double fAll = getCosSimfeatureAll(title, bodyParts.get(i));
						featureVec.add(fAll);

						entry.add(fAll.toString());

						fAll = getCosSimfeatureAll(title,
								bodyParts.get(1) + " " + bodyParts.get(i) + " " + bodyParts.get(3));
						featureVec.add(fAll);

						entry.add(fAll.toString());
					}
				}

				if (entry.size() != 12)
					throw new Exception("not 9 features");

				cosSims.put(title + s.get(1), featureVec);

				entries.add(entry.toArray(new String[0]));
			}
		}

		cosSims.save();
		cosSims.close();

		CSVWriter writer = new CSVWriter(new FileWriter(filepath + ".csv"));
		writer.writeAll(entries);
		writer.flush();
		writer.close();
		System.out.println("saved saved saved");

	}

	private Double getCosSimfeatureAll(String title, String body) {
		return calcDistance(title, body);
	}

	private List<Double> getCosSimfeaturePart(String title, String part, int partNo) {
		List<Double> subVec = new ArrayList<>();

		if (part.equals("")) {

			System.out.println("This cannot happen!");
			if (partNo == 1) {
				Double[] vec = new Double[BodySummarizerWithArguments.NUM_SENT_BEG];
				Arrays.fill(vec, -100.0);
				return Arrays.asList(vec);
			} else if (partNo == 3) {
				Double[] vec = new Double[BodySummarizerWithArguments.NUM_SENT_END];
				Arrays.fill(vec, -100.0);
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
				double f = calcDistance(title, s.toString());
				subVec.add(f);
			}

			if (partNo == 1)
				while (subVec.size() < BodySummarizerWithArguments.NUM_SENT_BEG)
					subVec.add(-100.0);

			if (partNo == 3)
				while (subVec.size() < BodySummarizerWithArguments.NUM_SENT_END)
					subVec.add(-100.0);
		}
		return subVec;
	}

	private double calcDistance(String title, String str2) {
		title = FeatureExtractorWithModifiedBL.getLemmatizedCleanStr(title);
		List<String> h = FeatureExtractorWithModifiedBL.removeStopWords(Arrays.asList(title.split("\\s+")));
		// get the string back
		StringBuilder sb = new StringBuilder();
		for (String s : h) {
			sb.append(s);
			sb.append(" ");
		}
		title = sb.toString().trim();

		str2 = FeatureExtractorWithModifiedBL.getLemmatizedCleanStr(str2);
		List<String> str2L = FeatureExtractorWithModifiedBL.removeStopWords(Arrays.asList(str2.split("\\s+")));
		// get the string back
		StringBuilder sb1 = new StringBuilder();
		for (String s : str2L) {
			sb1.append(s);
			sb1.append(" ");
		}
		str2 = sb1.toString().trim();

		// return cosSimMetric.distance(title, str2);
		return cosSimStringMetric.compare(title, str2);
	}

	public void getLeskOverlapFeatureVector(List<List<String>> stances,
			HashMap<Integer, Map<Integer, String>> summIdBoyMap, String filepath) throws Exception {
		if (lgo == null)
			initLesk();

		// HashMap to save values
		FileHashMap<String, ArrayList<Double>> leskOverlap = new FileHashMap<String, ArrayList<Double>>(filepath,
				FileHashMap.FORCE_OVERWRITE);

		List<String[]> entries = new ArrayList<>();
		List<String> csvHeader = new ArrayList<>();
		csvHeader.add("title");
		csvHeader.add("Body ID");
		csvHeader.add("Stance");

		for (int i = 0; i < 1; i++) {
			csvHeader.add("lesk_" + i);
		}

		entries.add(csvHeader.toArray(new String[0]));

		int i = 0;
		for (List<String> s : stances) {
			List<String> entry = new ArrayList<>();
			String title = s.get(0);
			entry.add(title);
			entry.add(s.get(1));
			entry.add(s.get(2));
			if (summIdBoyMap.containsKey(Integer.valueOf(s.get(1)))) {

				Map<Integer, String> bodyParts = summIdBoyMap.get(Integer.valueOf(s.get(1)));

				ArrayList<Double> featureVec = new ArrayList<>();
				//Double fAll = getLeskfeatureAll(title, bodyParts.get(1) + " " + bodyParts.get(3));
				Double fAll = getLeskfeatureAll(title, bodyParts.get(2));
				featureVec.add(fAll);

				entry.add(fAll.toString());

				if (entry.size() != 4)
					throw new Exception("not 1 feature");

				leskOverlap.put(title + s.get(1), featureVec);

				entries.add(entry.toArray(new String[0]));
			}
			
			if(i % 1000 == 0)
				System.out.println("Processed: " + i);
			i++;
		}

		leskOverlap.save();
		leskOverlap.close();

		CSVWriter writer = new CSVWriter(new FileWriter(filepath + ".csv"));
		writer.writeAll(entries);
		writer.flush();
		writer.close();
		System.out.println("saved saved saved");

	}

	private Double getLeskfeatureAll(String title, String body) {
		return lgo.overlap(title, body);
	}

	private List<Double> getLeskfeaturePart(String title, String part, int partNo) {
		List<Double> subVec = new ArrayList<>();

		if (part.equals("")) {
			System.out.println("This cannot happen!");
			if (partNo == 1) {
				Double[] vec = new Double[BodySummarizerWithArguments.NUM_SENT_BEG];
				Arrays.fill(vec, -100.0);
				return Arrays.asList(vec);
			} else if (partNo == 3) {
				Double[] vec = new Double[BodySummarizerWithArguments.NUM_SENT_END];
				Arrays.fill(vec, -100.0);
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
				double f = lgo.overlap(title, s.toString());
				subVec.add(f);
			}

			if (partNo == 1)
				while (subVec.size() < BodySummarizerWithArguments.NUM_SENT_BEG)
					subVec.add(-100.0);

			if (partNo == 3)
				while (subVec.size() < BodySummarizerWithArguments.NUM_SENT_END)
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

	public static FileHashMap<String, ArrayList<Double>> loadCosSimFeaturesAsHashFile(String hashFileName)
			throws FileNotFoundException, ObjectExistsException, ClassNotFoundException, VersionMismatchException,
			IOException {
		FileHashMap<String, ArrayList<Double>> cosSims = new FileHashMap<String, ArrayList<Double>>(hashFileName,
				FileHashMap.FORCE_OVERWRITE);
		return cosSims;
	}

	public static FileHashMap<String, ArrayList<Double>> loadLeskFeaturesAsHashFile(String hashFileName)
			throws FileNotFoundException, ObjectExistsException, ClassNotFoundException, VersionMismatchException,
			IOException {
		FileHashMap<String, ArrayList<Double>> leskOverlap = new FileHashMap<String, ArrayList<Double>>(hashFileName,
				FileHashMap.FORCE_OVERWRITE);
		return leskOverlap;
	}

	public static void loadData() throws IOException {
		StanceDetectionDataReader sddr = new StanceDetectionDataReader(true, true,
				ProjectPaths.TRAIN_STANCES_PREPROCESSED, ProjectPaths.ARGUMENTED_MID_BODIES33_TRAIN,
				ProjectPaths.TEST_STANCESS_PREPROCESSED, ProjectPaths.ARGUMENTED_MID_BODIES33_TEST);

		trainingSummIdBoyMap = sddr.readSummIdBodiesMap(new File(ProjectPaths.ARGUMENTED_MID_BODIES33_TRAIN));
		testSummIdBoyMap = sddr.readSummIdBodiesMap(new File(ProjectPaths.ARGUMENTED_MID_BODIES33_TEST));

		trainingStances = sddr.getTrainStances();

		testStances = sddr.getTestStances();
	}

	public static void main(String[] args) throws Exception {
		loadData();
		SimilarityFeatures33MidArguments sf = new SimilarityFeatures33MidArguments();
		//sf.getCosSimFeatureVector(trainingStances, trainingSummIdBoyMap, ProjectPaths.TRAIN_COS_SIM_STRMET_PARTS_ARG33_PATH);
		//sf.getCosSimFeatureVector(testStances, testSummIdBoyMap, ProjectPaths.TEST_COS_SIM_STRMET_PARTS_ARG33_PATH);

		sf.getLeskOverlapFeatureVector(trainingStances, trainingSummIdBoyMap, ProjectPaths.TRAIN_LESK_MID_PART_ARG33_PATH);
		sf.getLeskOverlapFeatureVector(testStances, testSummIdBoyMap, ProjectPaths.TEST_LESK_MID_PART_ARG33_PATH);
	}

}
