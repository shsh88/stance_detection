package ude.master.thesis.stance_detection.util;

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

public class BodySummarizerWithArguments {

	public static final int NUM_SENT_BEG = 5;
	public static final int NUM_SENT_END = 3;
	private Map<Integer, String> trainingIdBoyMap;
	private Map<Integer, String> testIdBoyMap;
	private List<List<String>> trainingStances;
	private List<List<String>> testStances;

	private StanfordCoreNLP pipeline;

	public BodySummarizerWithArguments() {
		Properties props;
		props = new Properties();
		props.put("annotators", "tokenize, ssplit, pos, lemma");

		this.pipeline = new StanfordCoreNLP(props);
	}

	public void loadData() throws IOException {
		StanceDetectionDataReader sddr = new StanceDetectionDataReader(true, true,
				ProjectPaths.TRAIN_STANCES_PREPROCESSED, ProjectPaths.PREPROCESSED_BODIES_TRAIN,
				ProjectPaths.TEST_STANCESS_PREPROCESSED, ProjectPaths.PREPROCESSED_BODIES_TEST);

		setTrainingIdBoyMap(sddr.getTrainIdBodyMap());
		setTestIdBoyMap(sddr.getTestIdBodyMap());

		setTrainingStances(sddr.getTrainStances());

		setTestStances(sddr.getTestStances());
	}

	public void summarize(Map<Integer, String> idBoyMap, String csvFilepath, String mapfilepath)
			throws FileNotFoundException, ObjectExistsException, ClassNotFoundException, VersionMismatchException,
			IOException {

		FileHashMap<Integer, Map<Integer, String>> summBodies = new FileHashMap<Integer, Map<Integer, String>>(
				mapfilepath, FileHashMap.FORCE_OVERWRITE);

		List<String[]> entries = new ArrayList<>();

		entries.add(new String[] { "Body ID", "sent_beg", "mid_part", "sent_end" });

		for (Entry<Integer, String> e : idBoyMap.entrySet()) {
			List<String> allSentences = new ArrayList<>();
			String body = e.getValue();

			Annotation doc = new Annotation(body);
			pipeline.annotate(doc);
			List<CoreMap> sentences = doc.get(SentencesAnnotation.class);

			if (sentences.size() >= 10) {
				for (CoreMap s : sentences) {
					allSentences.add(s.toString());
				}

				// get the different parts
				String begPart = "";
				begPart = getTextFromBeg(NUM_SENT_BEG, allSentences);

				String endPart = "";
				endPart = getTextFromEnd(NUM_SENT_END, allSentences);

				String midBody = getTextFromSetences(allSentences, NUM_SENT_BEG, NUM_SENT_END);

				// add to the file map
				Map<Integer, String> partsMap = new HashMap<>();
				partsMap.put(1, begPart);
				partsMap.put(2, midBody);
				partsMap.put(3, endPart);
				summBodies.put(e.getKey(), partsMap);

				List<String> entry = new ArrayList<>();
				entry.add(e.getKey().toString());
				entry.add(begPart);
				entry.add(midBody);
				entry.add(endPart);

				entries.add(entry.toArray(new String[0]));
			}
		}

		summBodies.save();
		summBodies.close();

		CSVWriter writer = new CSVWriter(new FileWriter(csvFilepath + ".csv"));
		writer.writeAll(entries);
		writer.flush();
		writer.close();
		System.out.println("saved saved saved");

	}

	private String getTextFromSetences(List<String> allSentences, int numSentBeg, int numSentEnd) {
		String part = "";
		for (int i = numSentBeg; i <= allSentences.size() - numSentEnd; i++) {
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

	/**
	 * No overlaps between first and last parts because we considered only 10
	 * sentences and more bodies
	 * 
	 * @param numSentEnd
	 * @param allSentences
	 * @return
	 */
	private String getTextFromEnd(int numSentEnd, List<String> allSentences) {
		String part = "";
		for (int i = allSentences.size() - numSentEnd; i < allSentences.size(); i++) {
			part += allSentences.get(i) + " ";
		}
		return part.trim();
	}

	public static void main(String[] args)
			throws IOException, ObjectExistsException, ClassNotFoundException, VersionMismatchException {
		BodySummarizerWithArguments bs = new BodySummarizerWithArguments();
		bs.loadData();
		bs.summarize(bs.getTrainingIdBoyMap(), ProjectPaths.SUMMARIZED_TRAIN_BODIES_PARTS_NOARGS,
				ProjectPaths.MAP_SUMMARIZED_TRAIN_BODIES_PARTS_NOARGS);
		bs.summarize(bs.getTestIdBoyMap(), ProjectPaths.SUMMARIZED_TEST_BODIES_PARTS_NOARGS,
				ProjectPaths.MAP_SUMMARIZED_TEST_BODIES_PARTS_NOARGS);
	}

	public Map<Integer, String> getTrainingIdBoyMap() {
		return trainingIdBoyMap;
	}

	public void setTrainingIdBoyMap(Map<Integer, String> trainingIdBoyMap) {
		this.trainingIdBoyMap = trainingIdBoyMap;
	}

	public Map<Integer, String> getTestIdBoyMap() {
		return testIdBoyMap;
	}

	public void setTestIdBoyMap(HashMap<Integer, String> testIdBoyMap) {
		this.testIdBoyMap = testIdBoyMap;
	}

	public List<List<String>> getTrainingStances() {
		return trainingStances;
	}

	public void setTrainingStances(List<List<String>> trainingStances) {
		this.trainingStances = trainingStances;
	}

	public List<List<String>> getTestStances() {
		return testStances;
	}

	public void setTestStances(List<List<String>> testStances) {
		this.testStances = testStances;
	}

}
