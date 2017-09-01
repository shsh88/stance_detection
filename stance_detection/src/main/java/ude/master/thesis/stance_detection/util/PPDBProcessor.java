package ude.master.thesis.stance_detection.util;

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

import ude.master.thesis.stance_detection.processor.FeatureExtractor;
import ude.master.thesis.stance_detection.processor.Lemmatizer;
import ude.master.thesis.stance_detection.processor.Porter;

public class PPDBProcessor {

	public static final double MIN_PPDB_SCORE = -10.0;
	public static final double MAX_PPDB_SCORE = 10.0;
	
	public static final String PPDB_2_S_Lexical = "C:/Master UDE/thesis/software/ppdb-2.0-s-lexical";
	public static final String MAP_PPDB_2_S_Lexical = "C:/Master UDE/thesis/software/mydata";
	
	public static final String PPDB_2_XXL_ALL = "C:/Master UDE/thesis/software/ppdb-2.0-xxl-all/ppdb-2.0-xxl-all";
	public static final String MAP_PPDB_2_XXL_ALL = "C:/Master UDE/thesis/software/ppdb-2.0-xxl-all/map_ppdb-2.0-xxl-all";
	
	
	static FileHashMap<String, ArrayList<ArrayList<String>>> ppdbData;
	private static Lemmatizer lemm;

	public static void main(String[] args) throws IOException {
		//extractParaphrases(PPDB_2_XXL_ALL, MAP_PPDB_2_XXL_ALL);

		// **************************************** read saved map

		// loadPPDB2("C:/Master UDE/thesis/software/mydata");
		//ppdbData = loadPPDB2("C:/Master UDE/thesis/software/mydata");
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

		// Generate tehe features from data and save them
		StanceDetectionDataReader sddr = new StanceDetectionDataReader(true, true, "resources/data/train_stances.csv",
				"resources/data/summ_train_bodies.csv", "resources/data/test_data/competition_test_stances.csv",
				"resources/data/test_data/summ_competition_test_bodies.csv");

		Map<Integer, String> trainIdBodyMap = sddr.getTrainIdBodyMap();
		List<List<String>> trainingStances = sddr.getTrainStances();
		//generateHungarianPPDBFeaturesAndSave(trainIdBodyMap, trainingStances,
				//"C:/thesis_stuff/features/train_hung_ppdb.csv");
		HashMap<Integer, String> testIdBodyMap = sddr.getTestIdBodyMap();
		List<List<String>> testStances = sddr.getTestStances();
		//generateHungarianPPDBFeaturesAndSave(testIdBodyMap, testStances, "C:/thesis_stuff/features/test_hung_ppdb.csv");
		saveHungarianScoreInFileMap("C:/thesis_stuff/features/test_features/test_hung_ppdb.csv", "C:/thesis_stuff/features/test_features/map_test_hung_ppdb");

		//Test saved Hungarian_Score Data
		//saveHungarianScoreInFileMap("C:/thesis_stuff/features/train_features/train_hung_ppdb.csv", "C:/thesis_stuff/features/train_features/map_train_hung_ppdb");
		//FileHashMap<String, ArrayList<Integer>> hung_scores = loadHungarianScoreFromFileMap("C:/thesis_stuff/features/train_features/map_train_hung_ppdb");
		//System.out.println(hung_scores.get("Banksy 'Arrested & Real Identity Revealed' Is The Same Hoax From Last Year1739"));
	}
	
	public static void saveHungarianScoreInFileMap(String txtFilename, String fileMapName){
		
		CSVReader reader = null;
		try {
			
			FileHashMap<String, ArrayList<Integer>> stanceData = new FileHashMap<String, ArrayList<Integer>>(
					fileMapName, FileHashMap.FORCE_OVERWRITE);
			
			reader = new CSVReader(new FileReader(txtFilename));
			String[] line;
			line = reader.readNext();
			
			while ((line = reader.readNext()) != null) {
				
				String idxStr = line[4];
				System.out.println(idxStr);
				ArrayList<Integer> idxs = getIntList(idxStr);
				
				stanceData.put(line[0]+line[1], idxs);
			}
			reader.close();
			
			//saving the map file
			stanceData.save();
			stanceData.close();
			
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
		idxStr = idxStr.substring(idxStr.indexOf('[')+1, idxStr.indexOf(']')).trim();
		String[] idxStrs = idxStr.split(", ");
		//create int Arraylist
		ArrayList<Integer> idxs = new ArrayList<>();
		for(String idx : idxStrs){
			idxs.add(Integer.valueOf(idx.trim()));
		}
		return idxs;
	}
	
	public static FileHashMap<String, ArrayList<Integer>> loadHungarianScoreFromFileMap(String filemapPath){
		FileHashMap<String, ArrayList<Integer>> stanceData = null;
		try {
			stanceData = new FileHashMap<String, ArrayList<Integer>>(filemapPath,
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

	private static void generateHungarianPPDBFeaturesAndSave(Map<Integer, String> idBodyMap, List<List<String>> stances,
			String filename) throws NumberFormatException, FileNotFoundException, IOException {
		List<String[]> entries = new ArrayList<>();
		entries.add(new String[] { "title", "Body ID", "Stance", "ppdb_hung_score", "indexes" });
		int i = 0;

		for (List<String> s : stances) {
			//System.out.println("stance = " + s);
			List<String> entry = new ArrayList<>();
			entry.add(s.get(0));
			entry.add(s.get(1));
			entry.add(s.get(2));

			ArrayList<Integer> indexes = new ArrayList<>();
			double score = calculateHungarianAlignmentScore(s.get(0), idBodyMap.get(Integer.valueOf(s.get(1))),
					indexes);

			entry.add(Double.toString(score));
			entry.add(indexes.toString());

			entries.add(entry.toArray(new String[0]));
			
			i++;
			//if (i == 10)
			//break;
			if (i % 10000 == 0)
				System.out.println("processed: " + i);
		}
		

		try (CSVWriter writer = new CSVWriter(new FileWriter(filename))) {
			writer.writeAll(entries);
		}

	}

	/**
	 * 
	 * @param header
	 * @param body
	 * @param indexes
	 *            the resulted indeceis
	 * @return
	 * @throws FileNotFoundException
	 * @throws IOException
	 */
	public static double calculateHungarianAlignmentScore(String header, String body, ArrayList<Integer> indexes)
			throws FileNotFoundException, IOException {
		String cleanH = FeatureExtractor.clean(header);
		String cleanB = FeatureExtractor.clean(body);

		List<String> hLemmas = lemm.lemmatize(cleanH);
		List<String> bLemmas = lemm.lemmatize(cleanB);

		hLemmas = FeatureExtractor.removeStopWords(hLemmas);
		bLemmas = FeatureExtractor.removeStopWords(bLemmas);
		
		//without it we get -1 in the resulting indexes
		while(bLemmas.size()< hLemmas.size())
			bLemmas.addAll(bLemmas);
		
		//System.out.println(hLemmas);
		//System.out.println(bLemmas);
		int i = 0;
		int j = 0;

		double[][] matrix = new double[hLemmas.size()][bLemmas.size()];

		for (String h : hLemmas) {
			j = 0;
			for (String b : bLemmas) {
				//System.out.print(h + " " + b);
				matrix[i][j] = computeParaphraseScore(h, b);
				//System.out.println(" " + matrix[i][j]);
				j++;
			}
			i++;
		}

		//printMat(matrix, hLemmas.size(), bLemmas.size());

		double[][] costMatrix = HungarianAlgorithm.makeCostMatrix(matrix, hLemmas.size(), bLemmas.size(),
				MAX_PPDB_SCORE);
		//printMat(costMatrix, hLemmas.size(), bLemmas.size());
		HungarianAlgorithm ha = new HungarianAlgorithm(costMatrix);
		//System.out.println("***");
		int[] idxs = ha.execute();
		//System.out.println("###");

		double total = 0.0;

		for (int k = 0; k < idxs.length; k++) {
			// for (int l = 0; l < idxs.length; l++) {
			//System.out.println("idxs[k] =  " + idxs[k]);
			//System.out.println(matrix[k][idxs[k]]);
			total += matrix[k][idxs[k]];
			// }

		}
		for (int g = 0; g < idxs.length; g++)
			indexes.add(idxs[g]);

		// DoubleStream stream =
		// Arrays.stream(matrix).flatMapToDouble(Arrays::stream);
		// double min = stream.min().getAsDouble();
		// System.out.println("min = "+min);

		return total / (double) (Math.min(hLemmas.size(), bLemmas.size()));
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

		//System.out.println("paraphrases1 = " + ppdbData.get(s));
		//System.out.println("paraphrases2 = " + ppdbData.get(sStem));
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
				
				if((data[1].split(" ").length > 1) || (data[2].split(" ").length > 1))
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

				/*if (textLhs.equals("neighbouring")) {
					System.out.println(textRhs + " " + ppdb2Score + " " + entailment);
					// System.out.println(str);
				}*/

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
			//System.out.println(ppdbScore.get("neighbouring"));
			//System.out.println(j);
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
