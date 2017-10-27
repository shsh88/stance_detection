package ude.master.thesis.stance_detection.featureengineering;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Map.Entry;

import org.clapper.util.misc.FileHashMap;
import org.clapper.util.misc.ObjectExistsException;
import org.clapper.util.misc.VersionMismatchException;

import com.opencsv.CSVWriter;

import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.util.CoreMap;
import ude.master.thesis.stance_detection.util.ProjectPaths;
import ude.master.thesis.stance_detection.util.StanceDetectionDataReader;

public class PunctuationCounter {

	private static StanfordCoreNLP pipeline;

	public static void main(String[] args)
			throws IOException, ObjectExistsException, ClassNotFoundException, VersionMismatchException {
		StanceDetectionDataReader sddr = new StanceDetectionDataReader(true, true,
				ProjectPaths.TRAIN_STANCES_PREPROCESSED, ProjectPaths.SUMMARIZED_TRAIN_BODIES2_WITH_MID,
				ProjectPaths.TEST_STANCESS_PREPROCESSED, ProjectPaths.SUMMARIZED_TEST_BODIES2_WITH_MID);

		List<List<String>> trainingStances = sddr.getTrainStances();
		HashMap<Integer, Map<Integer, String>> trainingSummIdBoyMap = sddr
				.readSummIdBodiesMap(new File(ProjectPaths.SUMMARIZED_TRAIN_BODIES2_WITH_MID));
		List<List<String>> testStances = sddr.getTestStances();
		HashMap<Integer, Map<Integer, String>> testSummIdBoyMap = sddr
				.readSummIdBodiesMap(new File(ProjectPaths.SUMMARIZED_TEST_BODIES2_WITH_MID));

		getBodiesPunctuation(trainingSummIdBoyMap, testSummIdBoyMap, ProjectPaths.PUNC_COUNT_TRAIN_TEST,
				ProjectPaths.CSV_PUNC_COUNT_TRAIN_TEST);

	}

	private static void getBodiesPunctuation(HashMap<Integer, Map<Integer, String>> trainingSummIdBoyMap,
			HashMap<Integer, Map<Integer, String>> testSummIdBoyMap, String hashFileName, String csvFileName)
			throws FileNotFoundException, ObjectExistsException, ClassNotFoundException, VersionMismatchException,
			IOException {
		FileHashMap<String, int[]> punc = new FileHashMap<String, int[]>(hashFileName, FileHashMap.FORCE_OVERWRITE);

		List<String[]> entries = new ArrayList<>();

		String[] header = new String[15];
		header[0] = "Body ID";
		header[1] = "Q_num";
		header[2] = "ex_num";

		entries.add(header);
		for (Entry<Integer, Map<Integer, String>> b : trainingSummIdBoyMap.entrySet()) {
			List<String> entry = new ArrayList<>();
			entry.add(b.getKey().toString());

			int[] puncList = new int[2];
			puncList[0] = 0;
			puncList[1] = 0;
			for (Entry<Integer, String> ins : b.getValue().entrySet()) {
				List<CoreMap> sentences = getSentencesFromParagraph(ins.getValue());
				for (CoreMap s : sentences) {
					if (s.toString().trim().endsWith("?"))
						puncList[0]++;
					if (s.toString().trim().endsWith("!"))
						puncList[1]++;
				}
			}
			punc.put(b.getKey().toString(), puncList);
			entry.add(String.valueOf(puncList[0]));
			entry.add(String.valueOf(puncList[1]));

			entries.add(entry.toArray(new String[0]));
		}

		for (Entry<Integer, Map<Integer, String>> b : testSummIdBoyMap.entrySet()) {
			List<String> entry = new ArrayList<>();
			entry.add(b.getKey().toString());

			int[] puncList = new int[2];
			puncList[0] = 0;
			puncList[1] = 0;
			for (Entry<Integer, String> ins : b.getValue().entrySet()) {
				List<CoreMap> sentences = getSentencesFromParagraph(ins.getValue());
				for (CoreMap s : sentences) {
					if (s.toString().trim().endsWith("?"))
						puncList[0]++;
					if (s.toString().trim().endsWith("!"))
						puncList[1]++;
				}
			}
			punc.put(b.getKey().toString(), puncList);
			entry.add(String.valueOf(puncList[0]));
			entry.add(String.valueOf(puncList[1]));

			entries.add(entry.toArray(new String[0]));
		}

		try (CSVWriter writer = new CSVWriter(new FileWriter(csvFileName))) {
			writer.writeAll(entries);
		}

		punc.save();
		punc.close();
	}

	private static List<CoreMap> getSentencesFromParagraph(String paragraph) {
		if (pipeline == null)
			initStanfordPipeline();

		Annotation doc = new Annotation(paragraph);
		pipeline.annotate(doc);
		List<CoreMap> sentences = doc.get(SentencesAnnotation.class);
		return sentences;
	}

	private static void initStanfordPipeline() {
		Properties props;
		props = new Properties();
		props.put("annotators", "tokenize, ssplit, pos, lemma");

		pipeline = new StanfordCoreNLP(props);

	}

	public static FileHashMap<String, int[]> loadPuncFeaturesAsHashFiles(String hashFileName)
			throws FileNotFoundException, ObjectExistsException, ClassNotFoundException, VersionMismatchException,
			IOException {
		FileHashMap<String, int[]> argsCountData = new FileHashMap<String, int[]>(hashFileName,
				FileHashMap.RECLAIM_FILE_GAPS);
		return argsCountData;
	}

}
