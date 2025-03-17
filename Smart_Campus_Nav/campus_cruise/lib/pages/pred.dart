import 'package:flutter/material.dart';
import 'package:webview_flutter/webview_flutter.dart';

// Loading Page
class LoadingPage extends StatelessWidget {
  const LoadingPage({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: const [
            CircularProgressIndicator(),
            SizedBox(height: 20),
            Text('Analyzing image...', style: TextStyle(fontSize: 18, fontWeight: FontWeight.w500)),
          ],
        ),
      ),
    );
  }
}

class PredictionResultPage extends StatefulWidget {
  final String landmarkName;
  final String location;
  final String busInfo;
  final String nearbyPlaces;

  const PredictionResultPage({
    Key? key,
    required this.landmarkName,
    required this.location,
    required this.busInfo,
    required this.nearbyPlaces,
  }) : super(key: key);

  @override
  State<PredictionResultPage> createState() => _PredictionResultPageState();
}

class _PredictionResultPageState extends State<PredictionResultPage> {
  late final WebViewController _controller;

  @override
  void initState() {
    super.initState();
    _controller = WebViewController()
      ..setJavaScriptMode(JavaScriptMode.unrestricted)
      ..loadRequest(Uri.parse(_getMapUrl(widget.landmarkName)));
  }

  // Get the map URL based on the prediction
  String _getMapUrl(String landmark) {
    Map<String, String> locationUrls = {
      "Academic Block A": "https://cins.iith.ac.in/?lat=17.5946&lon=78.1234",
      "Academic Block B": "https://cins.iith.ac.in/?lat=17.5962&lon=78.1265",
      "Academic Block C": "https://cins.iith.ac.in/?lat=17.5980&lon=78.1290",
    };
    return locationUrls[landmark] ?? "https://cins.iith.ac.in/";
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(backgroundColor: Colors.purple, title: const Text('Identify Landmark')),
      body: Column(
        children: [
          Expanded(
            child: WebViewWidget(controller: _controller),
          ),
          Container(
            color: Colors.black,
            padding: const EdgeInsets.all(20.0),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(widget.landmarkName, style: const TextStyle(color: Colors.white, fontSize: 22, fontWeight: FontWeight.bold)),
                Text('Your Location', style: const TextStyle(color: Colors.white70, fontSize: 14)),
                Text(widget.location, style: const TextStyle(color: Colors.white, fontSize: 16)),
                const Divider(color: Colors.white54),
              ],
            ),
          ),
        ],
      ),
    );
  }
}