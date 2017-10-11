package ude.master.thesis.stance_detection.analsys;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Set;

import com.opencsv.CSVWriter;

import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.util.CoreMap;
import ude.master.thesis.stance_detection.util.ProjectPaths;
import ude.master.thesis.stance_detection.util.StanceDetectionDataReader;

public class ArticleBodiesLengthAnalysis {

	private static Map<Integer, String> trainIdBodyMap = new HashMap<Integer, String>();
	private static List<List<String>> trainingStances = new ArrayList<>();
	private static HashMap<Integer, String> testIdBodyMap = new HashMap<>();
	private static List<List<String>> testStances = new ArrayList<List<String>>();
	private static HashMap<Integer, Map<Integer, String>> trainingSummIdBoyMap = new HashMap<>();
	private static HashMap<Integer, Map<Integer, String>> testSummIdBoyMap = new HashMap<>();
	
	private static List<List<String>> trainingStancesReduced = new ArrayList<>();
	private static List<List<String>> testStancesReduced = new ArrayList<List<String>>();

	StanfordCoreNLP pipeline;

	public static void loadData() throws IOException {
		StanceDetectionDataReader sddr = new StanceDetectionDataReader(true, true,
				ProjectPaths.TRAIN_STANCES_PREPROCESSED, ProjectPaths.SUMMARIZED_TRAIN_BODIES2,
				ProjectPaths.TEST_STANCESS_PREPROCESSED, ProjectPaths.SUMMARIZED_TEST_BODIES2);

		trainingSummIdBoyMap = sddr.readSummIdBodiesMap(new File(ProjectPaths.SUMMARIZED_TRAIN_BODIES2));
		testSummIdBoyMap = sddr.readSummIdBodiesMap(new File(ProjectPaths.SUMMARIZED_TEST_BODIES2));

		trainingStances = sddr.getTrainStances();
		trainingStancesReduced.addAll(trainingStances);

		testStances = sddr.getTestStances();
		testStancesReduced.addAll(testStances);
	}

	public void count(List<List<String>> stances, List<List<String>> stancesReduced, HashMap<Integer, Map<Integer, String>> summIdBoyMap,
			String dataType) {
		int lessThan10 = 0;
		List<String> lessThan10Data = new ArrayList<>();
		Set<String> lessThan10Ids = new HashSet<>();
		
		int equals9 = 0;
		List<String> equals9Data = new ArrayList<>();
		Set<String> equals9Ids = new HashSet<>();
		
		int equals8 = 0;
		List<String> equals8Data = new ArrayList<>();
		Set<String> equals8Ids = new HashSet<>();
		
		int equals7 = 0;
		List<String> equals7Data = new ArrayList<>();
		Set<String> equals7Ids = new HashSet<>();
		
		int equals6 = 0;
		List<String> equals6Data = new ArrayList<>();
		Set<String> equals6Ids = new HashSet<>();

		int lessThan5 = 0;
		List<String> lessThan5Data = new ArrayList<>();
		Set<String> lessThan5Ids = new HashSet<>();

		int lessThan2 = 0;
		List<String> lessThan2Data = new ArrayList<>();
		Set<String> lessThan2Ids = new HashSet<>();

		Map<Integer, ArrayList<String>> tLengthData = new HashMap<>();
		tLengthData.put(1, new ArrayList<>());
		tLengthData.put(2, new ArrayList<>());
		tLengthData.put(3, new ArrayList<>());
		Set<String> titles1 = new HashSet<>();
		Set<String> titles2 = new HashSet<>();
		Set<String> titles3 = new HashSet<>();

		List<List<String>> toRemove = new ArrayList<>();
		int i =0;
		for (List<String> s : stances) {
			Map<Integer, String> bodyParts = summIdBoyMap.get(Integer.valueOf(s.get(1)));

			// for (int i = 1; i <= 3; i++) {
			int c = countSentences(bodyParts.get(2));
			// }

			if (c < 10) {
				lessThan10++;
				lessThan10Data.add(s.get(0) + " , " + s.get(1) + " , " + s.get(2));
				lessThan10Ids.add(s.get(1));
			}
			if (c == 9) {
				equals9++;
				equals9Data.add(s.get(0) + " , " + s.get(1) + " , " + s.get(2));
				equals9Ids.add(s.get(1));
			}
			if (c == 8) {
				equals8++;
				equals8Data.add(s.get(0) + " , " + s.get(1) + " , " + s.get(2));
				equals8Ids.add(s.get(1));

			}
			if (c == 7) {
				equals7++;
				equals7Data.add(s.get(0) + " , " + s.get(1) + " , " + s.get(2));
				equals7Ids.add(s.get(1));
			}
			if (c == 6) {
				equals6++;
				equals6Data.add(s.get(0) + " , " + s.get(1) + " , " + s.get(2));
				equals6Ids.add(s.get(1));
			}
			if (c <= 5) {
				lessThan5++;
				lessThan5Data.add(s.get(0) + " , " + s.get(1) + " , " + s.get(2));
				lessThan5Ids.add(s.get(1));
			}
			if (c <= 2) {
				lessThan2++;
				lessThan2Data.add(s.get(0) + " , " + s.get(1) + " , " + s.get(2));
				lessThan2Ids.add(s.get(1));
				toRemove.add(stancesReduced.get(stances.indexOf(s)));
			}

			String title = s.get(0);
			int tLength = title.split("\\s+").length;

			if (tLength == 1) {
				tLengthData.get(1).add(s.get(0) + " , " + s.get(1) + " , " + s.get(2));
				titles1.add(s.get(0));
			}

			if (tLength == 2) {
				tLengthData.get(2).add(s.get(0) + " , " + s.get(1) + " , " + s.get(2));
				titles2.add(s.get(0));
			}

			if (tLength == 3) {
				tLengthData.get(3).add(s.get(0) + " , " + s.get(1) + " , " + s.get(2));
				titles3.add(s.get(0));
			}
			i++;
		}

		System.out.println("before: " + stancesReduced.size());
		stancesReduced.removeAll(toRemove);
		System.out.println("after: " + stancesReduced.size());
		
		Map<Integer, Set<String>> allTitles = new HashMap<>();
		allTitles.put(1, titles1);
		allTitles.put(2, titles2);
		allTitles.put(3, titles3);

		if (dataType.equals("train")) {
			writeInfo(10, lessThan10, lessThan10Data, lessThan10Ids, ProjectPaths.TRAIN_LESS_10_BODY);
			writeInfo(5, lessThan5, lessThan5Data, lessThan5Ids, ProjectPaths.TRAIN_LESS_5_BODY);
			writeInfo(2, lessThan2, lessThan2Data, lessThan2Ids, ProjectPaths.TRAIN_LESS_2_BODY);
			writeInfo(9, equals9, equals9Data, equals9Ids, ProjectPaths.TRAIN_eq9_BODY);
			writeInfo(8, equals8, equals8Data, equals8Ids, ProjectPaths.TRAIN_eq8_BODY);
			writeInfo(7, equals7, equals7Data, equals7Ids, ProjectPaths.TRAIN_eq7_BODY);
			writeInfo(6, equals6, equals6Data, equals6Ids, ProjectPaths.TRAIN_eq6_BODY);

			writeTitlesInfo(tLengthData, allTitles, ProjectPaths.TRAIN_TITLES_LENGTH_DATA);
		} else {
			writeInfo(10, lessThan10, lessThan10Data, lessThan10Ids, ProjectPaths.TEST_LESS_10_BODY);
			writeInfo(5, lessThan5, lessThan5Data, lessThan5Ids, ProjectPaths.TEST_LESS_5_BODY);
			writeInfo(2, lessThan2, lessThan2Data, lessThan2Ids, ProjectPaths.TEST_LESS_2_BODY);
			writeInfo(9, equals9, equals9Data, equals9Ids, ProjectPaths.TEST_eq9_BODY);
			writeInfo(8, equals8, equals8Data, equals8Ids, ProjectPaths.TEST_eq8_BODY);
			writeInfo(7, equals7, equals7Data, equals7Ids, ProjectPaths.TEST_eq7_BODY);
			writeInfo(6, equals6, equals6Data, equals6Ids, ProjectPaths.TEST_eq6_BODY);

			writeTitlesInfo(tLengthData, allTitles, ProjectPaths.TEST_TITLES_LENGTH_DATA);
		}
	}

	private void writeTitlesInfo(Map<Integer, ArrayList<String>> data, Map<Integer, Set<String>> allTitles,
			String path) {

		Writer writer = null;

		try {
			writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(path), "utf-8"));

			for (Map.Entry<Integer, ArrayList<String>> e : data.entrySet()) {
				writer.write("titles with length " + e.getKey() + "\n");
				writer.write(allTitles.get(e.getKey()).toString());
				for (String d : e.getValue()) {
					writer.write(d + "\n");
				}
			}
		} catch (IOException ex) {
			// report
		} finally {
			try {
				writer.close();
			} catch (Exception ex) {
				/* ignore */}
		}

	}

	private void writeInfo(int lessThancount, int count, List<String> instances, Set<String> lessThanIds, String path) {
		Writer writer = null;

		try {
			writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(path), "utf-8"));
			writer.write("Bodies that contain less than" + lessThancount + "sentences: " + count + "\n");

			writer.write("Bodies' ids: " + lessThanIds);
			for (String ins : instances) {
				writer.write(ins + "\n");
			}
		} catch (IOException ex) {
			// report
		} finally {
			try {
				writer.close();
			} catch (Exception ex) {
				/* ignore */}
		}
	}

	private int countSentences(String text) {
		if (pipeline == null)
			pipeline = getStanfordPipeline();

		Annotation doc = new Annotation(text);
		pipeline.annotate(doc);
		List<CoreMap> sentences = doc.get(SentencesAnnotation.class);

		return sentences.size();
	}

	private static StanfordCoreNLP getStanfordPipeline() {
		Properties props = new Properties();
		props.setProperty("annotators", "tokenize,ssplit,pos,lemma");
		StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
		return pipeline;
	}

	public static void main(String[] args) throws IOException {
		ArticleBodiesLengthAnalysis x = new ArticleBodiesLengthAnalysis();
		x.loadData();
		//x.count(trainingStances, trainingStancesReduced, trainingSummIdBoyMap, "train");
		//x.count(testStances, testStancesReduced, testSummIdBoyMap, "test");
		
		//saveCSV(trainingStancesReduced, ProjectPaths.TRAIN_STANCES_LESS2_PREPROCESSED);
		//saveCSV(testStancesReduced, ProjectPaths.TEST_STANCESS_LESS2_PREPROCESSED);
	}

	private static void saveCSV(List<List<String>> stances, String path) throws IOException {
		List<String[]> entries = new ArrayList<>();
		entries.add(new String[] { "Headline", "Body ID", "Stance" });
		
		for (List<String> stance : stances) {
			List<String> entry = new ArrayList<>();
			entry.add(stance.get(0));
			entry.add(stance.get(1));
			entry.add(stance.get(2));
			
			entries.add(entry.toArray(new String[0]));
			
		}
		
		CSVWriter writer = new CSVWriter(new FileWriter(path));
		writer.writeAll(entries);
		writer.flush();
		writer.close();
		System.out.println("saved saved saved");
	}

}
