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

public class PPDBProcessorWithArguments {

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
		ppdbData = loadPPDB2("C:/Master UDE/thesis/software/mydata");
		lemm = new Lemmatizer();
		/*
		 * //discuss -6 String headline =
		 * "ISIL Beheads American Photojournalist in Iraq"; String body =
		 * "James Foley, an American journalist who went missing in Syria more than a year ago, "
		 * +
		 * "has reportedly been executed by the Islamic State, a militant group formerly known as ISIS. "
		 * +
		 * "Video and photos purportedly of Foley emerged on Tuesday. A YouTube video -- entitled \"\"A "
		 * +
		 * "Message to #America (from the #IslamicState)\"\" -- identified a man on his knees as"
		 * +
		 * " \"\"James Wright Foley,\"\" and showed his execution. This is a developing story. "
		 * + "Check back here for updates.";
		 * 
		 * //agree 0.5279842857142855 String headline1 =
		 * "Banksy 'Arrested & Real Identity Revealed' Is The Same Hoax From Last Year"
		 * ; String body1 =
		 * "If you’ve seen a story floating around on your Facebook feed about Banksy getting "
		 * +
		 * "arrested and exposed, don’t worry because Banksy is still anonymous and well. "
		 * +
		 * "The hoax was the work of a “satirical” news site called The National Report — "
		 * +
		 * "which I’m not linking because f*ck those guys — that makes up fake stories that sound "
		 * +
		 * "like they could be ostensibly true, usually without a trace of actual satire, as filthy, "
		 * + "filthy clickbait. HAHA get it? It’s funny because it’s not true! "
		 * +
		 * "I guess there’s some logic in there somewhere. It’s not even entertaining to see people "
		 * +
		 * "share stories that come from places like the The National Report since they’re designed to "
		 * +
		 * "trick — unlike when dumb people share stories from The Onion, which are designed to "
		 * +
		 * "actually be satire. Again, and I cannot stress this enough: F*ck those guys."
		 * ; //unrelated -10 String headline2 =
		 * "Microsoft Tried Out Robot Security Guards on Its Silicon Valley Campus"
		 * ; String body2 =
		 * "A British rapper whose father is awaiting trial in Manhattan for a pair of US "
		 * +
		 * "embassy bombings is a leading suspect in the barbaric beheading of American journalist "
		 * +
		 * "James Foley, it was revealed on Friday. Abdel-Majed Abdel Bary — who recently tweeted a "
		 * +
		 * "photo of himself holding up a severed head — was among three Brits identified as possibly"
		 * +
		 * " being the masked killer known as “John the Beatle. But the brothers were released last year"
		 * +
		 * " after neither British journalist John Cantile nor Dutch photographer Jeroen Oerlmans "
		 * +
		 * "appeared to testify against them. Another suspect is Aine Davis, 30, a former drug "
		 * +
		 * "dealer and gang member who The Telegraph said converted to Islam and flew to Syria to "
		 * + "wage jihad.";
		 * 
		 * //unrelated -6 String headline3 =
		 * "British rapper a suspect in ISIS beheading"; String body3 =
		 * "Islamic State group fighters seized at least one cache of weapons airdropped by U.S.-led coalition forces that were meant to supply Kurdish militiamen "
		 * +
		 * "battling the extremist group in a border town, activists said today. The cache of "
		 * +
		 * "weapons included hand grenades, ammunition and rocket-propelled grenade launchers, "
		 * +
		 * "according to a video uploaded by a media group loyal to the Islamic State (ISIS) group. In "
		 * +
		 * "the short interview she alleged that she had been approached and accused of spying after "
		 * +
		 * "a report in which he said she claimed to have received images of Islamic State terrorists "
		 * +
		 * "being smuggled over the Turkey-Syria in vehicles belonging to the World Food Organization "
		 * +
		 * "and other aid groups. Shim described herself as 'surprised' at the accusation, "
		 * +
		 * "'because I have nothing to hide and I have never done anything aside my job.'"
		 * ; ArrayList<Integer> indexes = new ArrayList<>(); double score =
		 * calculateHungarianAlignmentScore(headline, body, indexes);
		 * System.out.println(score + "   " + indexes);
		 */

		// Generate the features from data and save them
		StanceDetectionDataReader sddr = new StanceDetectionDataReader(true, true,
				ProjectPaths.TRAIN_STANCES_PREPROCESSED, ProjectPaths.ARGUMENTED_BODIES_TRAIN,
				ProjectPaths.TEST_STANCESS_PREPROCESSED, ProjectPaths.ARGUMENTED_BODIES_TEST);


		List<List<String>> trainingStances = sddr.getTrainStances();
		Map<Integer, String> trainingIdBoyMap = sddr.getTrainIdBodyMap();
		//generateHungarianPPDBFeaturesAndSave(trainingIdBoyMap, trainingStances,
			//	ProjectPaths.CSV_SPPDB_HUNG_SCORES_IDXS_ARG_TRAIN);

		List<List<String>> testStances = sddr.getTestStances();

		HashMap<Integer, String> testIdBoyMap = sddr.getTestIdBodyMap();
		//generateHungarianPPDBFeaturesAndSave(testIdBoyMap, testStances,
			//	ProjectPaths.CSV_SPPDB_HUNG_SCORES_IDXS_ARG_TEST);
		
		//saveHungarianScoreInFileMap(ProjectPaths.CSV_SPPDB_HUNG_SCORES_IDXS_ARG_TRAIN,
			//	 ProjectPaths.SPPDB_HUNG_SCORES_IDXS_ARG_TRAIN);
		
		//saveHungarianScoreInFileMap(ProjectPaths.CSV_SPPDB_HUNG_SCORES_IDXS_ARG_TEST,
			//	ProjectPaths.SPPDB_HUNG_SCORES_IDXS_ARG_TEST);
		
		savePPDBFeaturesAsHashFiles(ProjectPaths.CSV_SPPDB_HUNG_SCORES_IDXS_ARG_TRAIN, ProjectPaths.SPPDB_HUNG_FEATURE_ARG_TRAIN);
		savePPDBFeaturesAsHashFiles(ProjectPaths.CSV_SPPDB_HUNG_SCORES_IDXS_ARG_TEST, ProjectPaths.SPPDB_HUNG_FEATURE_ARG_TEST);
		
		// Test saved Hungarian_Score Data
		// FileHashMap<String, ArrayList<Integer>> hung_scores =
		// loadHungarianScoreFromFileMap("C:/thesis_stuff/features/train_features/map_train_hung_ppdb");
		// System.out.println(hung_scores.get("Banksy 'Arrested & Real Identity
		// Revealed' Is The Same Hoax From Last Year1739"));
	}

	/**
	 * The score is just saved first for the sentences and at last for the whole body
	 * @param txtFilename
	 * @param fileMapName
	 */
	public static void saveHungarianScoreInFileMap(String txtFilename, String fileMapName) {

		CSVReader reader = null;
		try {

			FileHashMap<String, ArrayList<ArrayList<Integer>>> hungarianScoreData = new FileHashMap<String, ArrayList<ArrayList<Integer>>>(
					fileMapName, FileHashMap.FORCE_OVERWRITE);

			reader = new CSVReader(new FileReader(txtFilename));
			String[] line;
			line = reader.readNext();

			while ((line = reader.readNext()) != null) {
				ArrayList<ArrayList<Integer>> idxList = new ArrayList<>();
				for (int i = 3; i < line.length - 1; i++) {
					String idxStr = line[i].split("\\|")[1];
					//System.out.println(idxStr);
					ArrayList<Integer> idxs = getIntList(idxStr);
					idxList.add(idxs);
				}
				
				hungarianScoreData.put(line[0] + line[1], idxList);
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

	public static FileHashMap<String, ArrayList<ArrayList<Integer>>> loadHungarianScoreFromFileMap(String filemapPath) {
		FileHashMap<String, ArrayList<ArrayList<Integer>>> stanceData = null;
		try {
			stanceData = new FileHashMap<String, ArrayList<ArrayList<Integer>>>(filemapPath, FileHashMap.RECLAIM_FILE_GAPS);
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
	 * The resulting features are represented in a vector of length no_body_sentences + 1 (the alignment score 
	 * between the tile and each sentence in the body, plus the alignment between the title and the whole body) 
	 * here not just the score is saved but also the indecis 
	 * @param trainingIdBoyMap
	 * @param stances
	 * @param filename
	 * @throws Exception 
	 */
	private static void generateHungarianPPDBFeaturesAndSave(Map<Integer, String> trainingIdBoyMap,
			List<List<String>> stances, String filename)
			throws Exception {
		List<String[]> entries = new ArrayList<>();

		String[] header = new String[14];
		header[0] = "title";
		header[1] = "Body ID";
		header[2] = "Stance";

		header[3] = "sent_ppdb_hung_score_with_idexes";
		header[4] = "all_ppdb_hung_score_with_idexes";
		entries.add(header);

		int i = 0;

		for (List<String> s : stances) {
			// System.out.println("stance = " + s);
			List<String> entry = new ArrayList<>();
			entry.add(s.get(0));
			entry.add(s.get(1));
			entry.add(s.get(2));

			String body = trainingIdBoyMap.get(Integer.valueOf(s.get(1)));
			
			processBodyAsSentences(s.get(0), entry, body);
			processBodyAsAll(s.get(0), entry, body);
		
			
			entries.add(entry.toArray(new String[0]));

			i++;
			if (i % 1000 == 0)
				System.out.println("processed: " + i);
		}

		try (CSVWriter writer = new CSVWriter(new FileWriter(filename))) {
			writer.writeAll(entries);
		}

	}
	
	/**
	 * Here we save just the score without the indecies
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

	private static void processBodyAsAll(String title, List<String> entry, String body) throws FileNotFoundException, IOException {
		ArrayList<Integer> indexes = new ArrayList<>();
		double score = calculateHungarianAlignmentScore(title, body, indexes);
		entry.add(Double.toString(score) + "|" + indexes.toString());
	}

	private static void processBodyAsSentences(String title, List<String> entry, String body)
			throws FileNotFoundException, IOException {
		List<String> scoreList = new ArrayList<>();

		// get each sentence alone from the part
		Annotation doc = new Annotation(body);
		pipeline.annotate(doc);
		List<CoreMap> sentences = doc.get(SentencesAnnotation.class);

		for (CoreMap sent : sentences) {
			ArrayList<Integer> indexes = new ArrayList<>();
			double score = calculateHungarianAlignmentScore(title, sent.toString(), indexes);
	
			scoreList.add(Double.toString(score) + "|" + indexes.toString());
		}
		
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
			// System.out.println(ppdbScore.get("neighbouring"));
			// System.out.println(j);
			// System.out.println(entailments);
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
			// System.out.println(ppdbScoreReader.get("dialectical"));
			// System.out.println(ppdbScoreReader.get("neighbouring"));
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
