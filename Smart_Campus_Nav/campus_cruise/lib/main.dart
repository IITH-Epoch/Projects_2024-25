import 'package:flutter/material.dart';
import 'package:campus_cruise/pages/home.dart';
import 'package:campus_cruise/pages/cam.dart';
import 'package:camera/camera.dart';
import 'package:campus_cruise/pages/pred.dart';

List<CameraDescription> cameras = [];

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  cameras = await availableCameras();
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'CampusCruise',
      theme: ThemeData(
        primarySwatch: Colors.purple,
      ),
      initialRoute: '/',
      routes: {
        '/': (context) => const HomePage(),
        '/identify-landmark': (context) => IdentifyLandmarkPage(camera: cameras.first),
        '/loading': (context) => const LoadingPage(),
        '/prediction-result': (context) => PredictionResultPage(
          landmarkName: '',
          location: '',
          busInfo: '',
          nearbyPlaces: '',
        ),
      },
    );
  }
}
