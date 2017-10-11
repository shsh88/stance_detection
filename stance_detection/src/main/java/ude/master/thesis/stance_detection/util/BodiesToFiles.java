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

public class BodiesToFiles {

	private Map<Integer, String> trainIdBodyMap = new HashMap<Integer, String>();
	private static List<List<String>> trainingStances = new ArrayList<>();
	private HashMap<Integer, String> testIdBodyMap = new HashMap<>();
	private static List<List<String>> testStances = new ArrayList<List<String>>();
	private static Map<Integer, String> trainingIdBoyMapPreprocessed = new HashMap<>();
	private static HashMap<Integer, String> testSummIdBoyMapPreprocessed = new HashMap<>();
	
	private static Map<Integer, String> trainingIdBodyMapArgumented = new HashMap<>();
	private static HashMap<Integer, String> testIdBodyMapArgumented = new HashMap<>();

	private StanfordCoreNLP pipeline;

	public BodiesToFiles() {
		Properties props;
		props = new Properties();
		props.put("annotators", "tokenize, ssplit, pos, lemma");

		this.pipeline = new StanfordCoreNLP(props);
	}

	public void loadData() throws IOException {
		StanceDetectionDataReader sddr = new StanceDetectionDataReader(true, true,
				ProjectPaths.TRAIN_STANCES_PREPROCESSED, ProjectPaths.PREPROCESSED_BODIES_TRAIN,
				ProjectPaths.TEST_STANCESS_PREPROCESSED, ProjectPaths.PREPROCESSED_BODIES_TEST);

		trainingIdBoyMapPreprocessed = sddr.getTrainIdBodyMap();
		testSummIdBoyMapPreprocessed = sddr.getTestIdBodyMap();

		trainingStances = sddr.getTrainStances();

		testStances = sddr.getTestStances();
	}

	public void saveBodiesFiles(String bodiescSVFilePath) throws IOException {
		CSVReader reader = new CSVReader(new FileReader(bodiescSVFilePath));
		String[] line;
		line = reader.readNext();

		while ((line = reader.readNext()) != null) {
			String fileName = line[0];
			List<String> bodySentences = getSentencesList(line[1]);
			saveInFile(fileName, bodySentences);

		}
		reader.close();
	}

	private void saveInFile(String fileName, List<String> bodySentences) throws IOException {
		Path file = Paths.get("C:/thesis_stuff/arguments_data/" + fileName + ".txt");
		Files.write(file, bodySentences, Charset.forName("UTF-8"));
	}

	private List<String> getSentencesList(String body) {
		Annotation doc = new Annotation(body);
		this.pipeline.annotate(doc);
		List<CoreMap> sentences = doc.get(SentencesAnnotation.class);

		List<String> allSentences = new ArrayList<>();
		for (CoreMap s : sentences) {
			if(s.toString().length()>2)
				allSentences.add(s.toString());
		}

		return allSentences;
	}
	
	public void readArgumentedBodiesFromFiles() throws IOException{
		File[] files = new File(ProjectPaths.ARGUMENTED_BODIES_FILES).listFiles();
		//If this pathname does not denote a directory, then listFiles() returns null. 

		for (File file : files) {
			String fileName = file.getName();
			Integer bodyId = Integer.valueOf(fileName.substring(0, fileName.lastIndexOf(".txt")));
			String bodyText = getArgumentedTextFromCSV(file);
		     if(testSummIdBoyMapPreprocessed.containsKey(bodyId)){
		    	 System.out.println("here");
		    	 testIdBodyMapArgumented.put(bodyId, bodyText);
		     }else
		    	 if(trainingIdBoyMapPreprocessed.containsKey(bodyId)){
		    		 System.out.println("hereeee");
		    		 trainingIdBodyMapArgumented.put(bodyId, bodyText);
		    	 }
		}
		System.out.println(testIdBodyMapArgumented.size());
		System.out.println(trainingIdBodyMapArgumented.size());
		saveIdBoyMapInCSV(trainingIdBodyMapArgumented, ProjectPaths.ARGUMENTED_BODIES_TRAIN);
		saveIdBoyMapInCSV(testIdBodyMapArgumented, ProjectPaths.ARGUMENTED_BODIES_TEST);
	}

	private void saveIdBoyMapInCSV(Map<Integer, String> idBoyMapArgumented, String path) throws IOException {
		List<String[]> entries = new ArrayList<>();
		entries.add(new String[] { "Body ID", "ArticleBody" });
		
		for (Entry<Integer, String> e : idBoyMapArgumented.entrySet()) {
			List<String> entry = new ArrayList<>();
			entry.add(String.valueOf(e.getKey()));
			entry.add(e.getValue());
			
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
			if(line[8].equals("Argument")){
				bodyText += line[0].trim() + " ";
			}
		}
		reader.close();
		return bodyText.trim();
	}

	public static void main(String[] args) throws IOException {
		BodiesToFiles bf = new BodiesToFiles();
		bf.loadData();
		//bf.saveBodiesFiles(ProjectPaths.PREPROCESSED_BODIES_TRAIN);
		//bf.saveBodiesFiles(ProjectPaths.PREPROCESSED_BODIES_TEST);
		bf.readArgumentedBodiesFromFiles();
	}
}
