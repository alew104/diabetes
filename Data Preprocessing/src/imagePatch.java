import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Random;
import java.util.Scanner;

import javax.imageio.ImageIO;

public class imagePatch {
	static class Marking {
		public int x;
		public int y;
		public String type;
		
		public Marking(String type, int x, int y) {
			// TODO Auto-generated constructor stub
			this.type = type;
			this.x = x;
			this.y = y;
		}
		
		public Marking() {
			
		}
		
		public String toString() {
			return "type = " + this.type + ", x = " + x + ", y = " + y;
		}
	}
	final static int WIDTH = 224;
	final static int HEIGHT = 224;
	final static String model = "VGG16";
	final static String fileDirectory = "/Users/alex/Desktop/" + model + "/";
	final static String hDirectory = "hemorrhages/image0";
	final static String nDirectory = "not_hemorrhages/image0";
	final static String hemorrhages = "hemorrhages/";
	final static String notHemorrhages = "not_hemorrhages/";
	
	
	public static void main(String[] args) throws IOException {
		// TODO Auto-generated method stub
		makeFolders();
		PrintWriter pw = new PrintWriter(new FileOutputStream(new File(fileDirectory + "labels.csv"), true));
		pw.println("filename, hemorrhages");
		
		for (int i = 1; i < 90; i++) {
			ArrayList<Marking> al = parseCoords(i);
			parseImage(i, al);
		}
		
		removeSmallImages(fileDirectory + "training/" + hemorrhages);
		removeSmallImages(fileDirectory + "validation/" + hemorrhages);
		removeSmallImages(fileDirectory + "testing/" + hemorrhages);
		
		removeSmallImages(fileDirectory + "training/" + notHemorrhages);
		removeSmallImages(fileDirectory + "validation/" + notHemorrhages);
		removeSmallImages(fileDirectory + "testing/" + notHemorrhages);
	}
	
	public static ArrayList<Marking> parseCoords(int imageNum) throws FileNotFoundException {
		String file;
		if (imageNum < 10) {
			file = "0" + imageNum; 
		} else {
			file = "" + imageNum;
		}
		ArrayList<Marking> al = new ArrayList<>();
		for (int i = 1; i < 5; i++) {
			File f = new File("out/diaretdb1_image0" + file + "_0" + i + ".txt");
			Scanner s = new Scanner(f);
			int x = 0;
			int y = 0;
			String type = "";
			while (s.hasNextLine()) {
				String[] values = s.nextLine().split(",");
				if (values[0].equals("markingtype")) {
					type = values[1];
					al.add(new Marking(type, x, y));
				} else if (values[0].equals("coords2d")) {
					x = Integer.parseInt(values[1]);
					y = Integer.parseInt(values[2]);
				}
			}
			s.close();
		}
		
		/*
		for (int i = 0; i < al.size(); i++) {
			System.out.println(al.get(i).toString());
		}
		*/
		return al;
	}
	
	public static void makeFolders() throws IOException {
		File files = new File(fileDirectory);
		files.mkdirs();
		files = new File(fileDirectory + "labels.csv");
		files.createNewFile();
		files = new File(fileDirectory + "training/" + hemorrhages);
		files.mkdirs();
		files = new File(fileDirectory + "testing/" + hemorrhages);
		files.mkdirs();
		files = new File (fileDirectory + "validation/" + hemorrhages);
		files.mkdirs();
		files = new File(fileDirectory + "training/" + notHemorrhages);
		files.mkdirs();
		files = new File(fileDirectory + "testing/" + notHemorrhages);
		files.mkdirs();
		files = new File (fileDirectory + "validation/" + notHemorrhages);
		files.mkdirs();
	}
	
	public static void parseImage(int imageNum, ArrayList<Marking> al) throws IOException {
		String file;
		if (imageNum < 10) {
			file = "0" + imageNum; 
		} else {
			file = "" + imageNum;
		}
		boolean f = new File("/Users/alex/Desktop/ddb1_fundusimages_out/image0" + file).mkdirs();
		PrintWriter pw = new PrintWriter(new FileOutputStream(new File(fileDirectory + "labels.csv"), true));
		final BufferedImage source = ImageIO.read(new File("/Users/alex/Desktop/images/diaretdb1_image0" + file + ".png"));
		int idx = 0;
		for (int y = 0; y < source.getHeight(); y += HEIGHT) {
			for (int x = 0; x < source.getWidth(); x += WIDTH) {
				boolean isHemorrhage = false;
				for (int i = 0; i < al.size(); i++) {
					int markingX = al.get(i).x;
					int markingY = al.get(i).y;
					if (((x < markingX) && (markingX < (x + WIDTH))) && ((y < markingY) && (markingY < (y + HEIGHT)))) {
						// write to file the block number
						if (al.get(i).type.equals("Haemorrhages")) {
							isHemorrhage = true;
						} 
					}
				}
				
				String filename;
				String folder = null;
				
				Random rand = new Random();
				int random = rand.nextInt(3);
				if (random == 0) {
					folder = "training/";
				} else if (random == 1) {
					folder = "validation/";
				} else if (random == 2) {
					folder = "testing/";
				}
				
				if (isHemorrhage) {
					filename = fileDirectory + folder + hDirectory + file + "_" + idx + ".png";
					String line = folder + "image0" + file + "_" + idx + ", " + "yes";
					pw.println(line);
				} else {
					filename = fileDirectory + folder + nDirectory + file + "_" + idx + ".png";
					String line = folder + "image0" + file + "_" + idx + ", " + "no";
					pw.println(line);
				}

				if ((x + WIDTH) > source.getWidth() && (y + HEIGHT) < source.getHeight()) {
					ImageIO.write(source.getSubimage(x, y, source.getWidth() - x, HEIGHT), "png", new File(filename));
				} else if ((y + HEIGHT) > source.getHeight() && (x + WIDTH < source.getWidth())){
					ImageIO.write(source.getSubimage(x, y, WIDTH, source.getHeight() - y), "png", new File(filename));
				} else if ((x + WIDTH > source.getWidth()) && (y + HEIGHT > source.getHeight())){
					ImageIO.write(source.getSubimage(x, y, source.getWidth() - x, source.getHeight() - y), "png", new File(filename));
				} else if (((x + WIDTH) < source.getWidth()) && ((y + HEIGHT) < source.getHeight()) ){
					ImageIO.write(source.getSubimage(x, y, WIDTH, HEIGHT), "png", new File(filename));
				}
				idx++;
			}
		}
		pw.close();
	}
	
	public static void removeSmallImages(String folderName) throws IOException {
		File folder = new File(folderName);
		File[] listOfFiles = folder.listFiles();
		for (File f : listOfFiles) {
			BufferedImage bimg = ImageIO.read(f);
			int width = bimg.getWidth();
			int height = bimg.getHeight();
			if (width < WIDTH ) {
				f.delete();
			} else if (height < HEIGHT) {
				f.delete();
			} else if (height < HEIGHT && width < WIDTH) {
				f.delete();
			}
		}
	}
}
