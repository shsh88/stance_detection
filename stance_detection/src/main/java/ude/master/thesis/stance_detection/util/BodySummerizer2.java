package ude.master.thesis.stance_detection.util;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import org.clapper.util.misc.FileHashMap;
import org.clapper.util.misc.ObjectExistsException;
import org.clapper.util.misc.VersionMismatchException;

import com.opencsv.CSVWriter;

import java.util.Properties;

import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.util.CoreMap;

public class BodySummerizer2 {

	public static final int NUM_SENT_BEG = 5;
	public static final int NUM_SENT_END = 3;

	private Map<Integer, String> trainIdBodyMap = new HashMap<Integer, String>();
	private List<List<String>> trainingStances = new ArrayList<>();
	private HashMap<Integer, String> testIdBodyMap = new HashMap<>();
	private List<List<String>> testStances = new ArrayList<List<String>>();
	private HashMap<Integer, Map<Integer, String>> trainingSummIdBoyMap = new HashMap<>();
	private HashMap<Integer, Map<Integer, String>> testSummIdBoyMap = new HashMap<>();
	private StanfordCoreNLP pipeline;

	public BodySummerizer2() {
		Properties props;
		props = new Properties();
		props.put("annotators", "tokenize, ssplit, pos, lemma");

		this.pipeline = new StanfordCoreNLP(props);
	}

	public void loadData() throws IOException {
		StanceDetectionDataReader sddr = new StanceDetectionDataReader(true, true,
				ProjectPaths.TRAIN_STANCES_PREPROCESSED, ProjectPaths.SUMMARIZED_TRAIN_BODIES,
				ProjectPaths.TEST_STANCESS_PREPROCESSED, ProjectPaths.SUMMARIZED_TEST_BODIES);

		setTrainingSummIdBoyMap(sddr.readSummIdBodiesMap(new File(ProjectPaths.SUMMARIZED_TRAIN_BODIES)));
		setTestSummIdBoyMap(sddr.readSummIdBodiesMap(new File(ProjectPaths.SUMMARIZED_TEST_BODIES)));

		trainingStances = sddr.getTrainStances();

		testStances = sddr.getTestStances();
	}

	public void summarize(HashMap<Integer, Map<Integer, String>> summIdBoyMap, String filepath, String mapfilepath)
			throws FileNotFoundException, ObjectExistsException, ClassNotFoundException, VersionMismatchException,
			IOException {

		FileHashMap<Integer, Map<Integer, String>> summBodies = new FileHashMap<Integer, Map<Integer, String>>(
				mapfilepath, FileHashMap.FORCE_OVERWRITE);

		List<String[]> entries = new ArrayList<>();

		entries.add(new String[] { "Body ID", "sent_beg", "all_body", "sent_end" });

		for (Entry<Integer, Map<Integer, String>> e : summIdBoyMap.entrySet()) {
			List<String> allSentences = new ArrayList<>();

			for (Entry<Integer, String> parts : e.getValue().entrySet()) {
				String part = parts.getValue();

				if (!part.isEmpty()) {
					Annotation doc = new Annotation(part);
					pipeline.annotate(doc);
					List<CoreMap> sentences = doc.get(SentencesAnnotation.class);
					for (CoreMap s : sentences) {
						allSentences.add(s.toString());
					}
				}
			}

			// get the different parts
			String begPart = "";
			if (allSentences.size() < NUM_SENT_BEG) {
				begPart = getTextFromSetences(allSentences);
			} else {
				begPart = getTextFromBeg(NUM_SENT_BEG, allSentences);
			}

			String endPart = "";
			if (allSentences.size() < NUM_SENT_END) {
				endPart = getTextFromSetences(allSentences);
			} else {
				endPart = getTextFromEnd(NUM_SENT_END, allSentences);
			}

			String allBody = getTextFromSetences(allSentences);

			// add to the file map
			Map<Integer, String> partsMap = new HashMap<>();
			partsMap.put(1, begPart);
			partsMap.put(2, allBody);
			partsMap.put(3, endPart);
			summBodies.put(e.getKey(), partsMap);

			List<String> entry = new ArrayList<>();
			entry.add(e.getKey().toString());
			entry.add(begPart);
			entry.add(allBody);
			entry.add(endPart);

			entries.add(entry.toArray(new String[0]));
		}

		summBodies.save();
		summBodies.close();

		CSVWriter writer = new CSVWriter(new FileWriter(filepath + ".csv"));
		writer.writeAll(entries);
		writer.flush();
		writer.close();
		System.out.println("saved saved saved");

	}

	private String getTextFromEnd(int numSentEnd, List<String> allSentences) {
		String part = "";
		for (int i = allSentences.size() - numSentEnd; i < allSentences.size(); i++) {
			part += allSentences.get(i) + " ";
		}
		return part.trim();
	}

	private String getTextFromBeg(int numSentBeg, List<String> allSentences) {
		String part = "";
		for (int i = 0; i < numSentBeg; i++) {
			part += allSentences.get(i) + " ";
		}
		return part.trim();
	}

	private String getTextFromSetences(List<String> allSentences) {
		String part = "";
		for (String sent : allSentences) {
			part += sent + " ";
		}
		return part.trim();
	}

	public static void main(String[] args)
			throws IOException, ObjectExistsException, ClassNotFoundException, VersionMismatchException {
		BodySummerizer2 bs = new BodySummerizer2();
		bs.loadData();
		bs.summarize(bs.getTrainingSummIdBoyMap(), ProjectPaths.SUMMARIZED_TRAIN_BODIES2,
				ProjectPaths.MAP_SUMMARIZED_TRAIN_BODIES2);
		bs.summarize(bs.getTestSummIdBoyMap(), ProjectPaths.SUMMARIZED_TEST_BODIES2,
				ProjectPaths.MAP_SUMMARIZED_TEST_BODIES2);
	}

	public HashMap<Integer, Map<Integer, String>> getTrainingSummIdBoyMap() {
		return trainingSummIdBoyMap;
	}

	public void setTrainingSummIdBoyMap(HashMap<Integer, Map<Integer, String>> trainingSummIdBoyMap) {
		this.trainingSummIdBoyMap = trainingSummIdBoyMap;
	}

	public HashMap<Integer, Map<Integer, String>> getTestSummIdBoyMap() {
		return testSummIdBoyMap;
	}

	public void setTestSummIdBoyMap(HashMap<Integer, Map<Integer, String>> testSummIdBoyMap) {
		this.testSummIdBoyMap = testSummIdBoyMap;
	}
}
