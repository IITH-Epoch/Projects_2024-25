import 'dart:io';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:tflite_flutter/tflite_flutter.dart' as tfl;
import 'package:image/image.dart' as img;
import 'package:campus_cruise/pages/pred.dart';

class IdentifyLandmarkPage extends StatefulWidget {
  final CameraDescription camera;

  const IdentifyLandmarkPage({Key? key, required this.camera}) : super(key: key);

  @override
  _IdentifyLandmarkPageState createState() => _IdentifyLandmarkPageState();
}

class _IdentifyLandmarkPageState extends State<IdentifyLandmarkPage> {
  late CameraController _controller;
  late Future<void> _initializeControllerFuture;
  late tfl.Interpreter _interpreter;
  String _landmarkPrediction = "No prediction yet";
  List<String> _classLabels = [];

  @override
  void initState() {
    super.initState();
    _controller = CameraController(
      widget.camera,
      ResolutionPreset.high,
      enableAudio: false,
    );
    _initializeControllerFuture = _controller.initialize();
    _loadModel();
  }

  Future<void> _loadModel() async {
    try {
      _interpreter = await tfl.Interpreter.fromAsset('assets/model.tflite');
      print('Model loaded successfully');

      // Load class labels
      String labelsString = await DefaultAssetBundle.of(context).loadString('assets/labels.txt');
      _classLabels = labelsString.split('\n').map((e) => e.trim()).where((e) => e.isNotEmpty).toList();

      print('Labels loaded: $_classLabels');
    } catch (e) {
      print('Error loading model or labels: $e');
    }
  }


  @override
  void dispose() {
    _controller.dispose();
    _interpreter.close();
    super.dispose();
  }

  /// Capture image and navigate to the loading screen before processing
  Future<void> _navigateToLoadingPage() async {
    try {
      await _initializeControllerFuture;
      final image = await _controller.takePicture();

      // Navigate to LoadingPage
      Navigator.push(
        context,
        MaterialPageRoute(builder: (context) => const LoadingPage()),
      );

      // Start the prediction process in the background
      await _predictLandmark(image.path);
    } catch (e) {
      print("Error capturing image: $e");
    }
  }

  /// Run inference and transition to `PredictionResultPage`
  Future<void> _predictLandmark(String imagePath) async {
    // Load and process image
    File imageFile = File(imagePath);
    List<int> imageBytes = await imageFile.readAsBytes();
    img.Image? image = img.decodeImage(imageBytes);

    if (image == null) {
      print("Unable to decode image");
      return;
    }

    // Resize the image to match model input size (100x100)
    img.Image resizedImage = img.copyResize(image, width: 100, height: 100);

    // Convert image to a normalized tensor
    List<List<List<List<double>>>> input = List.generate(
      1,
          (_) => List.generate(
        100,
            (_) => List.generate(100, (_) => List.filled(3, 0.0)),
      ),
    );

    for (int y = 0; y < 100; y++) {
      for (int x = 0; x < 100; x++) {
        int pixel = resizedImage.getPixel(x, y);
        input[0][y][x][0] = img.getRed(pixel) / 255.0;
        input[0][y][x][1] = img.getGreen(pixel) / 255.0;
        input[0][y][x][2] = img.getBlue(pixel) / 255.0;
      }
    }

    // Define output buffer based on number of labels
    var output = List.generate(1, (_) => List.filled(_classLabels.length, 0.0));

    // Run inference
    _interpreter.run(input, output);

    // Find the highest probability index
    int predictedIndex = output[0].indexOf(output[0].reduce((a, b) => a > b ? a : b));
    String predictedLabel = _classLabels[predictedIndex];

    setState(() {
      _landmarkPrediction = predictedLabel;
    });

    // Transition to `PredictionResultPage`
    Navigator.pushReplacement(
      context,
      MaterialPageRoute(
        builder: (context) => PredictionResultPage(
          landmarkName: predictedLabel,
          location: "Sample Location",
          busInfo: "Bus information",
          nearbyPlaces: "Nearby places",
        ),
      ),
    );
  }


  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Identify Landmark'),
        backgroundColor: Colors.purple,
      ),
      body: FutureBuilder<void>(
        future: _initializeControllerFuture,
        builder: (context, snapshot) {
          if (snapshot.connectionState == ConnectionState.done) {
            return Column(
              children: [
                Expanded(child: CameraPreview(_controller)),
                Container(
                  color: Colors.black,
                  padding: const EdgeInsets.symmetric(vertical: 10.0),
                  child: Row(
                    mainAxisAlignment: MainAxisAlignment.spaceAround,
                    children: [
                      IconButton(
                        icon: const Icon(Icons.flip_camera_ios, color: Colors.white, size: 30),
                        onPressed: () {}, // Implement camera switching if needed
                      ),
                      GestureDetector(
                        onTap: _navigateToLoadingPage, // Start process
                        child: Container(
                          height: 70,
                          width: 70,
                          decoration: const BoxDecoration(color: Colors.white, shape: BoxShape.circle),
                        ),
                      ),
                      const SizedBox(width: 48),
                    ],
                  ),
                ),
              ],
            );
          } else {
            return const Center(child: CircularProgressIndicator());
          }
        },
      ),
    );
  }
}