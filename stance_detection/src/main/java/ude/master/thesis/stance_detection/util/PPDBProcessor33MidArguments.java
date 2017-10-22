package ude.master.thesis.stance_detection.util;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.LineNumberReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.clapper.util.misc.FileHashMap;
import org.clapper.util.misc.ObjectExistsException;
import org.clapper.util.misc.VersionMismatchException;

import com.opencsv.CSVReader;
import com.opencsv.CSVWriter;

import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.util.CoreMap;
import ude.master.thesis.stance_detection.processor.FeatureExtractor;
import ude.master.thesis.stance_detection.processor.Lemmatizer;
import ude.master.thesis.stance_detection.processor.Porter;
import ude.master.thesis.stance_detection.processor.StanfordDependencyParser;

public class PPDBProcessor33MidArguments {

	public static final double MIN_PPDB_SCORE = -10.0;
	public static final double MAX_PPDB_SCORE = 10.0;

	public static final String PPDB_2_S_Lexical = "C:/Master UDE/thesis/software/ppdb-2.0-s-lexical";
	public static final String MAP_PPDB_2_S_Lexical = "C:/Master UDE/thesis/software/mydata";

	public static final String PPDB_2_XXL_ALL = "C:/Master UDE/thesis/software/ppdb-2.0-xxl-all/ppdb-2.0-xxl-all";
	public static final String MAP_PPDB_2_XXL_ALL = "C:/Master UDE/thesis/software/ppdb-2.0-xxl-all/map_ppdb-2.0-xxl-all";

	static FileHashMap<String, ArrayList<ArrayList<String>>> ppdbData;
	private static Lemmatizer lemm;
	private static StanfordCoreNLP pipeline;

	public static void main(String[] args) throws Exception {
		pipeline = TitleAndBodyTextPreprocess.getStanfordPipeline();
		// extractParaphrases(PPDB_2_XXL_ALL, MAP_PPDB_2_XXL_ALL);

		// **************************************** read saved map

		// loadPPDB2("C:/Master UDE/thesis/software/mydata");
		ppdbData = loadPPDB2(MAP_PPDB_2_S_Lexical);
		lemm = new Lemmatizer();

		// Generate the features from data and save them
		StanceDetectionDataReader sddr = new StanceDetectionDataReader(true, true,
				ProjectPaths.TRAIN_STANCES_PREPROCESSED, ProjectPaths.ARGUMENTED_MID_BODIES33_TRAIN,
				ProjectPaths.TEST_STANCESS_PREPROCESSED, ProjectPaths.ARGUMENTED_MID_BODIES33_TEST);

		List<List<String>> trainingStances = sddr.getTrainStances();
		HashMap<Integer, Map<Integer, String>> trainingSummIdBoyMap = sddr
				.readSummIdBodiesMap(new File(ProjectPaths.ARGUMENTED_MID_BODIES33_TRAIN));
		//generateHungarianPPDBFeaturesAndSave(trainingSummIdBoyMap, trainingStances,
			//	ProjectPaths.CSV_PPDB_HUNG_SCORES_IDXS_PARTS_ARG33_TRAIN);

		List<List<String>> testStances = sddr.getTestStances();

		HashMap<Integer, Map<Integer, String>> testSummIdBoyMap = sddr
				.readSummIdBodiesMap(new File(ProjectPaths.ARGUMENTED_MID_BODIES33_TEST));
		//generateHungarianPPDBFeaturesAndSave(testSummIdBoyMap, testStances,
			//	ProjectPaths.CSV_PPDB_HUNG_SCORES_IDXS_PARTS_ARG33_TEST);

		saveHungarianScoreIdxsInFileMap(ProjectPaths.CSV_PPDB_HUNG_SCORES_IDXS_PARTS_ARG33_TRAIN,
				ProjectPaths.PPDB_HUNG_SCORES_IDXS_PARTS_ARG33_TRAIN);

		saveHungarianScoreIdxsInFileMap(ProjectPaths.CSV_PPDB_HUNG_SCORES_IDXS_PARTS_ARG33_TEST,
				ProjectPaths.PPDB_HUNG_SCORES_IDXS_PARTS_ARG33_TEST);

		 savePPDBFeaturesAsHashFiles(ProjectPaths.CSV_PPDB_HUNG_SCORES_IDXS_PARTS_ARG33_TRAIN,
				 ProjectPaths.PPDB_HUNG_FEATURE_PARTS_ARG33_TRAIN);
		 savePPDBFeaturesAsHashFiles(ProjectPaths.CSV_PPDB_HUNG_SCORES_IDXS_PARTS_ARG33_TEST,
				 ProjectPaths.PPDB_HUNG_FEATURE_PARTS_ARG33_TEST);

		// Test saved Hungarian_Score Data
		// FileHashMap<String, ArrayList<Integer>> hung_scores =
		// loadHungarianScoreFromFileMap("C:/thesis_stuff/features/train_features/map_train_hung_ppdb");
		// System.out.println(hung_scores.get("Banksy 'Arrested & Real Identity
		// Revealed' Is The Same Hoax From Last Year1739"));
	}

	public static void saveHungarianScoreIdxsInFileMap(String txtFilename, String fileMapName) {

		CSVReader reader = null;
		try {

			FileHashMap<String, Map<Integer, ArrayList<ArrayList<Integer>>>> hungarianScoreData = new FileHashMap<String, Map<Integer, ArrayList<ArrayList<Integer>>>>(
					fileMapName, FileHashMap.FORCE_OVERWRITE);

			reader = new CSVReader(new FileReader(txtFilename));
			String[] line;
			line = reader.readNext();
			Map<Integer, ArrayList<ArrayList<Integer>>> idxsMap = new HashMap<>();

			while ((line = reader.readNext()) != null) {
				ArrayList<ArrayList<Integer>> idxListP1 = new ArrayList<>();
				for (int i = 3; i < 7; i++) {
					String idxStr = line[i].split("\\|")[1];
					// System.out.println(idxStr);
					ArrayList<Integer> idxs = getIntList(idxStr);
					idxListP1.add(idxs);
				}
				idxsMap.put(1, idxListP1);
				
				ArrayList<ArrayList<Integer>> idxListP2 = new ArrayList<>();
				for (int i = 11; i < line.length; i++) {
					String idxStrAll = line[i].split("\\|")[1];
					// System.out.println(idxStr);
					ArrayList<Integer> idxsAll = getIntList(idxStrAll);
					
					idxListP2.add(idxsAll);
				}
				idxsMap.put(2, idxListP2);

				ArrayList<ArrayList<Integer>> idxListP3 = new ArrayList<>();
				for (int i = 7; i < 11; i++) {
					String idxStr = line[i].split("\\|")[1];
					// System.out.println(idxStr);
					ArrayList<Integer> idxs = getIntList(idxStr);
					idxListP3.add(idxs);
				}
				idxsMap.put(3, idxListP3);
				hungarianScoreData.put(line[0] + line[1], idxsMap);
			}
			reader.close();

			// saving the map file
			hungarianScoreData.save();
			hungarianScoreData.close();

		} catch (IOException e) {
			e.printStackTrace();
		} catch (ObjectExistsException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (ClassNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (VersionMismatchException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}

	public static ArrayList<Integer> getIntList(String idxStr) {
		if (idxStr.equals("[]"))
			return new ArrayList<>();
		idxStr = idxStr.substring(idxStr.indexOf('[') + 1, idxStr.indexOf(']')).trim();
		String[] idxStrs = idxStr.split(", ");
		// create int Arraylist
		ArrayList<Integer> idxs = new ArrayList<>();
		for (String idx : idxStrs) {
			idxs.add(Integer.valueOf(idx.trim()));
		}
		return idxs;
	}

	public static FileHashMap<String, Map<Integer, ArrayList<ArrayList<Integer>>>> loadHungarianScoreIdxsFromFileMap(
			String filemapPath) {
		FileHashMap<String, Map<Integer, ArrayList<ArrayList<Integer>>>> stanceData = null;
		try {
			stanceData = new FileHashMap<String, Map<Integer, ArrayList<ArrayList<Integer>>>>(filemapPath,
					FileHashMap.RECLAIM_FILE_GAPS);
		} catch (ObjectExistsException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (ClassNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (VersionMismatchException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return stanceData;
	}

	/**
	 * The resulting features are represented in a vector of length 11 (the
	 * alignment score between the tile and each sentence in the body; 5
	 * sentences from the beginning and 3 from the end. Plus the alignment
	 * between the title and the whole body) here not just the score is saved
	 * but also the indecis
	 * 
	 * @param summIdBoyMap
	 * @param stances
	 * @param filename
	 * @throws Exception
	 */
	private static void generateHungarianPPDBFeaturesAndSave(HashMap<Integer, Map<Integer, String>> summIdBoyMap,
			List<List<String>> stances, String filename) throws Exception {
		List<String[]> entries = new ArrayList<>();

		String[] header = new String[14];
		header[0] = "title";
		header[1] = "Body ID";
		header[2] = "Stance";

		for (int j = 1; j <= 8; j++) {
			if (j >= 1 && j <= 3) {
				header[j + 2] = "begin_ppdb_hung_score_with_idexes" + j;
			}
			if (j == 4) {
				header[j + 2] = "beginpart_ppdb_hung_score_with_idexes";
			}
			if (j >= 5 && j <= 7) {
				header[j + 2] = "end_ppdb_hung_score_with_idexes" + j;
			}
			if (j == 8) {
				header[j + 2] = "endpart_ppdb_hung_score_with_idexes";
			}
		}
		header[9 + 2] = "mid_ppdb_hung_score_with_idexes";
		header[10 + 2] = "midSent_ppdb_hung_score_with_idexes"; // this is not just one value
		header[11 + 2] = "midSent_ppdb_hung_score_with_idexes";
		entries.add(header);

		int i = 0;

		for (List<String> s : stances) {
			// System.out.println("stance = " + s);
			List<String> entry = new ArrayList<>();
			entry.add(s.get(0));
			entry.add(s.get(1));
			entry.add(s.get(2));
			if (summIdBoyMap.containsKey(Integer.valueOf(s.get(1)))) {
				Map<Integer, String> bodyParts = summIdBoyMap.get(Integer.valueOf(s.get(1)));
				for (int k = 1; k <= 3; k++) {// ***
					String part = bodyParts.get(k);
					if (k != 2) {
						processPart(s.get(0), entry, part, k);
					} 
				}
				//System.out.println("before: " + entry.size());
				for (int k = 1; k <= 3; k++){
					if (k == 2) {
						String part = bodyParts.get(k);
						processPartAsAll(s.get(0), entry, part);
						processMidPartAsSentences(s.get(0), entry, part);
					}
				}
				
				processPartAsAll(s.get(0), entry, bodyParts.get(1) + " " + bodyParts.get(2) + " " + bodyParts.get(3));
				
				//System.out.println(entry.size());
				//if (entry.size() != 14)
					//throw new Exception("not 11 values");

				entries.add(entry.toArray(new String[0]));
			}
			i++;
			// if (i == 10)
			// break;
			if (i % 1000 == 0)
				System.out.println("processed: " + i);
		}

		try (CSVWriter writer = new CSVWriter(new FileWriter(filename))) {
			writer.writeAll(entries);
		}

	}
	/**
	 * 
	 * @param title
	 * @param entry
	 * @param part
	 * @throws FileNotFoundException
	 * @throws IOException
	 */
	private static void processMidPartAsSentences(String title, List<String> entry, String part) throws FileNotFoundException, IOException {
		List<String> scoreList = new ArrayList<>();

		// get each sentence alone from the part
		Annotation doc = new Annotation(part);
		pipeline.annotate(doc);
		List<CoreMap> sentences = doc.get(SentencesAnnotation.class);

		for (CoreMap sent : sentences) {
			ArrayList<Integer> indexes = new ArrayList<>();
			double score = calculateHungarianAlignmentScore(title, sent.toString(), indexes);

			scoreList.add(Double.toString(score) + "|" + indexes.toString());
		}

		ArrayList<Integer> partIndexes = new ArrayList<>();
		double partScore = calculateHungarianAlignmentScore(title, part, partIndexes);
		scoreList.add(Double.toString(partScore) + "|" + partIndexes.toString());

		entry.addAll(scoreList);
		
	}

	/**
	 * Here we save just the score without the indecies
	 * 
	 * @param csvfilePath
	 * @param hashFileName
	 * @throws FileNotFoundException
	 * @throws ObjectExistsException
	 * @throws ClassNotFoundException
	 * @throws VersionMismatchException
	 * @throws IOException
	 */
	public static void savePPDBFeaturesAsHashFiles(String csvfilePath, String hashFileName)
			throws FileNotFoundException, ObjectExistsException, ClassNotFoundException, VersionMismatchException,
			IOException {
		FileHashMap<String, double[]> ppdbData = new FileHashMap<String, double[]>(hashFileName,
				FileHashMap.FORCE_OVERWRITE);
		CSVReader reader = null;
		reader = new CSVReader(new FileReader(csvfilePath));
		String[] line;
		line = reader.readNext();

		int lineNo = 0;

		while ((line = reader.readNext()) != null) {
			double[] featureVecor = new double[line.length - 3];
			for (int i = 3; i < line.length; i++) {
				Double v = Double.valueOf(line[i].split("\\|")[0]);
				featureVecor[i - 3] = v;
			}
			ppdbData.put(line[0] + line[1], featureVecor);
			lineNo++;
			if (lineNo % 1000 == 0)
				System.out.println(lineNo);
		}
		reader.close();

		// saving the map file
		ppdbData.save();
		ppdbData.close();

	}

	public static FileHashMap<String, double[]> loadPPDBFeaturesAsHashFiles(String hashFileName)
			throws FileNotFoundException, ObjectExistsException, ClassNotFoundException, VersionMismatchException,
			IOException {
		FileHashMap<String, double[]> ppdbData = new FileHashMap<String, double[]>(hashFileName,
				FileHashMap.RECLAIM_FILE_GAPS);
		return ppdbData;
	}

	private static void processPartAsAll(String title, List<String> entry, String part)
			throws FileNotFoundException, IOException {
		if (part.isEmpty())
			entry.add(Double.toString(-100.0) + "|" + "[]");
		else {
			ArrayList<Integer> indexes = new ArrayList<>();
			double score = calculateHungarianAlignmentScore(title, part, indexes);
			entry.add(Double.toString(score) + "|" + indexes.toString());
		}
	}

	private static void processPart(String title, List<String> entry, String part, int partNo)
			throws FileNotFoundException, IOException {
		List<String> scoreList = new ArrayList<>();

		// get each sentence alone from the part
		Annotation doc = new Annotation(part);
		pipeline.annotate(doc);
		List<CoreMap> sentences = doc.get(SentencesAnnotation.class);

		for (CoreMap sent : sentences) {
			ArrayList<Integer> indexes = new ArrayList<>();
			double score = calculateHungarianAlignmentScore(title, sent.toString(), indexes);

			scoreList.add(Double.toString(score) + "|" + indexes.toString());
		}

		if (partNo == 1) {
			while (scoreList.size() < BodySummarizerWithArguments.NUM_SENT_BEG)
				scoreList.add(Double.toString(-100.0) + "|" + "[]");
		} else {
			if (partNo == 3) {
				while (scoreList.size() < BodySummarizerWithArguments.NUM_SENT_END)
					scoreList.add(Double.toString(-100.0) + "|" + "[]");
			}
		}

		ArrayList<Integer> partIndexes = new ArrayList<>();
		double partScore = calculateHungarianAlignmentScore(title, part, partIndexes);
		scoreList.add(Double.toString(partScore) + "|" + partIndexes.toString());

		entry.addAll(scoreList);
	}

	/**
	 * 
	 * @param header
	 * @param bodySentence
	 * @param indexes
	 *            the resulted indeceis
	 * @return
	 * @throws FileNotFoundException
	 * @throws IOException
	 */
	public static double calculateHungarianAlignmentScore(String header, String bodySentence,
			ArrayList<Integer> indexes) throws FileNotFoundException, IOException {

		List<String> hLemmas = lemm.lemmatize(header);
		List<String> bSentLemmas = lemm.lemmatize(bodySentence);

		// hLemmas = FeatureExtractor.removeStopWords(hLemmas);
		// bSentLemmas = FeatureExtractor.removeStopWords(bSentLemmas);

		// System.out.println(hLemmas);
		// System.out.println(bLemmas);
		int i = 0;
		int j = 0;

		double[][] matrix = new double[hLemmas.size()][bSentLemmas.size()];

		for (String h : hLemmas) {
			j = 0;
			for (String b : bSentLemmas) {
				// System.out.print(h + " " + b);
				matrix[i][j] = computeParaphraseScore(h, b);
				// System.out.println(" " + matrix[i][j]);
				j++;
			}
			i++;
		}

		// printMat(matrix, hLemmas.size(), bLemmas.size());

		double[][] costMatrix = HungarianAlgorithm.makeCostMatrix(matrix, hLemmas.size(), bSentLemmas.size(),
				MAX_PPDB_SCORE);
		// printMat(costMatrix, hLemmas.size(), bLemmas.size());
		HungarianAlgorithm ha = new HungarianAlgorithm(costMatrix);
		// System.out.println("***");
		int[] idxs = ha.execute();
		// System.out.println("###");

		double total = 0.0;

		for (int k = 0; k < idxs.length; k++) {
			// for (int l = 0; l < idxs.length; l++) {
			// System.out.println("idxs[k] = " + idxs[k]);
			// System.out.println(matrix[k][idxs[k]]);
			if (idxs[k] != -1)
				total += matrix[k][idxs[k]];
			// }

		}
		for (int g = 0; g < idxs.length; g++)
			if (idxs[g] != -1)
				indexes.add(idxs[g]);

		// DoubleStream stream =
		// Arrays.stream(matrix).flatMapToDouble(Arrays::stream);
		// double min = stream.min().getAsDouble();
		// System.out.println("min = "+min);

		return total / (double) (Math.min(hLemmas.size(), bSentLemmas.size()));
	}

	public static void printMat(double mat[][], int n, int m) {
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < m; j++)
				System.out.print(mat[i][j] + " ");
			System.out.println();
		}
	}

	public static double computeParaphraseScore(String s, String t) throws FileNotFoundException, IOException {
		String sStem = stem(s);
		String tStem = stem(t);

		if (sStem.equals(tStem)) {
			return MAX_PPDB_SCORE;
		}

		// get PPDB paraphrases of s, and find matches to t, up to stemming
		// System.out.println("s= " + s);
		List<ArrayList<String>> sParaphrases = new ArrayList<>();

		if (ppdbData.get(s) != null)
			sParaphrases.addAll(ppdbData.get(s));

		// System.out.println("sStem= " + sStem + " " + ppdbData.get(sStem));
		if (ppdbData.get(sStem) != null)
			sParaphrases.addAll(ppdbData.get(sStem));

		// System.out.println("paraphrases1 = " + ppdbData.get(s));
		// System.out.println("paraphrases2 = " + ppdbData.get(sStem));
		boolean matchFound = false;
		double maxScore = -10.0;

		for (ArrayList<String> p : sParaphrases) {
			if ((p.get(0).equals(t)) || (p.get(0).equals(tStem))) {
				matchFound = true;
				if (Double.valueOf(p.get(1)) > maxScore)
					maxScore = Double.valueOf(p.get(1));
			}

		}
		if (matchFound)
			return maxScore;

		return MIN_PPDB_SCORE;
	}

	private static String stem(String str) {
		Porter porter = new Porter();
		return porter.stripAffixes(str);
	}

	/**
	 * Saves PPDB data into a readable FileHashMap
	 * 
	 * @param ppdbOriginalFilename
	 * @param extractedPPDBFilename
	 * @throws IOException
	 */
	private static void extractParaphrases(String ppdbOriginalFilename, String extractedPPDBFilename)
			throws IOException {
		FileReader fr = null;
		LineNumberReader lnr = null;
		String str;
		int i;

		try {

			// create new reader
			fr = new FileReader(ppdbOriginalFilename);
			lnr = new LineNumberReader(fr);

			FileHashMap<String, ArrayList<ArrayList<String>>> ppdbScore = new FileHashMap<String, ArrayList<ArrayList<String>>>(
					extractedPPDBFilename, FileHashMap.FORCE_OVERWRITE);

			// read lines till the end of the stream
			int j = 0;
			// Set<String> entailments = new HashSet<>();
			while ((str = lnr.readLine()) != null) {
				i = lnr.getLineNumber();
				// System.out.print("("+i+")");

				// prints string
				// System.out.println(str);

				String[] data = str.split(" \\|\\|\\| ");

				if ((data[1].split(" ").length > 1) || (data[2].split(" ").length > 1))
					continue;

				String textLhs = data[1];
				String textRhs = data[2];

				// System.out.println(data[3]);
				// System.out.println(data[3].split(" ")[1]);
				// System.out.println(data[3].split(" ")[0]);
				double ppdb2Score = Double.valueOf(data[3].split(" ")[0].split("=")[1]);
				String entailment = data[data.length - 1].trim();

				// entailments.add(entailment);

				String key = textLhs;
				ArrayList<String> value = new ArrayList<>();
				value.add(textRhs);
				value.add(Double.toString(ppdb2Score));
				value.add(entailment);

				/*
				 * if (textLhs.equals("neighbouring")) {
				 * System.out.println(textRhs + " " + ppdb2Score + " " +
				 * entailment); // System.out.println(str); }
				 */

				// if (entailment.contains("ReverseEntailment"))
				// System.out.println(textLhs + " " + textRhs);
				// System.out.println(textLhs + " " + textRhs + " " + ppdb2Score
				// + " " + entailment);

				j++;

				// if(j==3)
				// break;
				if (ppdbScore.containsKey(key)) {
					ArrayList<ArrayList<String>> values = ppdbScore.get(key);
					values.add(value);
					ppdbScore.put(key, values);

				} else {
					ArrayList<ArrayList<String>> paphrases = new ArrayList<>();
					paphrases.add(value);
					ppdbScore.put(key, paphrases);
				}

			}
			ppdbScore.save();
			ppdbScore.close();
		} catch (Exception e) {

			// if any error occurs
			e.printStackTrace();
		} finally {

			// closes the stream and releases system resources
			if (fr != null)
				fr.close();
			if (lnr != null)
				lnr.close();
		}
	}

	public static FileHashMap<String, ArrayList<ArrayList<String>>> loadPPDB2(String path)
			throws FileNotFoundException, IOException {
		FileHashMap<String, ArrayList<ArrayList<String>>> ppdbScoreReader = null;
		try {
			ppdbScoreReader = new FileHashMap<String, ArrayList<ArrayList<String>>>(path,
					FileHashMap.RECLAIM_FILE_GAPS);

		} catch (ObjectExistsException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (ClassNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (VersionMismatchException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return ppdbScoreReader;
	}
}
