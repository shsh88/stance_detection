package ude.master.thesis.stance_detection.util;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Properties;

import com.opencsv.CSVReader;
import com.opencsv.CSVWriter;

import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.util.CoreMap;

public class MidBodiesToFiles {

	private static List<List<String>> trainingStances = new ArrayList<>();
	private static List<List<String>> testStances = new ArrayList<List<String>>();

	private StanfordCoreNLP pipeline;
	private HashMap<Integer, Map<Integer, String>> testSummIdBoyMap;
	private HashMap<Integer, Map<Integer, String>> trainingSummIdBoyMap;

	private HashMap<Integer, Map<Integer, String>> testSummArgIdBoyMap;
	private HashMap<Integer, Map<Integer, String>> trainingSummArgIdBoyMap;

	public MidBodiesToFiles() {
		Properties props;
		props = new Properties();
		props.put("annotators", "tokenize, ssplit, pos, lemma");

		this.pipeline = new StanfordCoreNLP(props);
	}

	public void loadData() throws IOException {
		StanceDetectionDataReader sddr = new StanceDetectionDataReader(true, true,
				ProjectPaths.TRAIN_STANCES_PREPROCESSED, ProjectPaths.SUMMARIZED_TRAIN_BODIES_PARTS_NOARGS,
				ProjectPaths.TEST_STANCESS_PREPROCESSED, ProjectPaths.SUMMARIZED_TEST_BODIES_PARTS_NOARGS);

		trainingSummIdBoyMap = sddr.readSummIdBodiesMap(new File(ProjectPaths.SUMMARIZED_TRAIN_BODIES_PARTS_NOARGS));
		testSummIdBoyMap = sddr.readSummIdBodiesMap(new File(ProjectPaths.SUMMARIZED_TEST_BODIES_PARTS_NOARGS));

		testSummArgIdBoyMap = new HashMap<>();
		trainingSummArgIdBoyMap = new HashMap<>();
		
		trainingStances = sddr.getTrainStances();

		testStances = sddr.getTestStances();
	}

	public void saveBodiesFiles(String bodiesCSVFilePath) throws IOException {
		CSVReader reader = new CSVReader(new FileReader(bodiesCSVFilePath));
		String[] line;
		line = reader.readNext();

		while ((line = reader.readNext()) != null) {
			String fileName = line[0];
			List<String> bodySentences = getSentencesList(line[2]);
			saveInFile(fileName, bodySentences);

		}
		reader.close();
	}

	private void saveInFile(String fileName, List<String> midBodySentences) throws IOException {
		Path file = Paths.get("C:/thesis_stuff/arguments_data/" + fileName + ".txt");
		Files.write(file, midBodySentences, Charset.forName("UTF-8"));
	}

	private List<String> getSentencesList(String body) {
		Annotation doc = new Annotation(body);
		this.pipeline.annotate(doc);
		List<CoreMap> sentences = doc.get(SentencesAnnotation.class);

		List<String> allSentences = new ArrayList<>();
		for (CoreMap s : sentences) {
			if (s.toString().length() > 2)
				allSentences.add(s.toString());
		}

		return allSentences;
	}

	public void readArgumentedBodiesPartsFromFiles() throws IOException {
		File[] files = new File(ProjectPaths.ARGUMENTED_BODIES_FILES).listFiles();
		// If this pathname does not denote a directory, then listFiles()
		// returns null.

		for (File file : files) {
			String fileName = file.getName();
			Integer bodyId = Integer.valueOf(fileName.substring(0, fileName.lastIndexOf(".txt")));
			String midBodyText = getArgumentedTextFromCSV(file);
			if (testSummIdBoyMap.containsKey(bodyId)) {
				Map<Integer, String> parts = new HashMap<>();
				parts.put(1, testSummArgIdBoyMap.get(bodyId).get(1));
				parts.put(2, midBodyText);
				parts.put(3, testSummArgIdBoyMap.get(bodyId).get(3));
				testSummArgIdBoyMap.put(bodyId, parts);
			} else if (trainingSummIdBoyMap.containsKey(bodyId)) {
				Map<Integer, String> parts = new HashMap<>();
				parts.put(1, trainingSummArgIdBoyMap.get(bodyId).get(1));
				parts.put(2, midBodyText);
				parts.put(3, trainingSummArgIdBoyMap.get(bodyId).get(3));
				trainingSummArgIdBoyMap.put(bodyId, parts);
			}
		}
		System.out.println(testSummArgIdBoyMap.size());
		System.out.println(trainingSummArgIdBoyMap.size());
		saveIdBoyMapInCSV(trainingSummArgIdBoyMap, ProjectPaths.ARGUMENTED_MID_BODIES_TRAIN);
		saveIdBoyMapInCSV(testSummArgIdBoyMap, ProjectPaths.ARGUMENTED_MID_BODIES_TEST);
	}

	private void saveIdBoyMapInCSV(HashMap<Integer, Map<Integer, String>> summArgIdMidBoyMap, String path)
			throws IOException {
		List<String[]> entries = new ArrayList<>();
		entries.add(new String[] { "Body ID", "sent_beg", "mid_arg_part", "sent_end" });

		for (Entry<Integer, Map<Integer, String>> e : summArgIdMidBoyMap.entrySet()) {
			List<String> entry = new ArrayList<>();
			entry.add(String.valueOf(e.getKey()));
			for (Entry<Integer, String> p : e.getValue().entrySet()) {
				entry.add(p.getValue());
			}
			entries.add(entry.toArray(new String[0]));

		}

		CSVWriter writer = new CSVWriter(new FileWriter(path));
		writer.writeAll(entries);
		writer.flush();
		writer.close();
		System.out.println("saved saved saved");
	}

	private String getArgumentedTextFromCSV(File file) throws IOException {
		CSVReader reader = new CSVReader(new FileReader(file.getPath()));
		String[] line;
		line = reader.readNext();
		String bodyText = "";
		while ((line = reader.readNext()) != null) {
			if (line[8].equals("Argument")) {
				bodyText += line[0].trim() + " ";
			}
		}
		reader.close();
		return bodyText.trim();
	}

	public static void main(String[] args) throws IOException {
		MidBodiesToFiles bf = new MidBodiesToFiles();
		bf.loadData();
		bf.saveBodiesFiles(ProjectPaths.SUMMARIZED_TRAIN_BODIES_PARTS_NOARGS);
		bf.saveBodiesFiles(ProjectPaths.SUMMARIZED_TEST_BODIES_PARTS_NOARGS);
		//bf.readArgumentedBodiesPartsFromFiles();
	}
}
