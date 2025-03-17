import 'package:flutter/material.dart';

class HomePage extends StatelessWidget {
  const HomePage({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      body: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        crossAxisAlignment: CrossAxisAlignment.center,
        children: [
          Center(
            child: Column(
              children: [
                // Logo Section
                CircleAvatar(
                  radius: 50,
                  backgroundColor: Colors.white,
                  child: Icon(
                    Icons.explore,
                    size: 50,
                    color: Colors.purple,
                  ),
                ),
                const SizedBox(height: 20),

                // App Title Section
                Text(
                  'CampusCruise',
                  style: TextStyle(
                    color: Colors.purple,
                    fontSize: 32,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                const SizedBox(height: 10),

                // Subtitle Section
                Text(
                  'The Smart Campus Navigation\nApp of IITH',
                  textAlign: TextAlign.center,
                  style: TextStyle(
                    color: Colors.white,
                    fontSize: 16,
                  ),
                ),
                const SizedBox(height: 40),

                // Identify Landmark Button
                ElevatedButton.icon(
                  onPressed: () {
                    Navigator.pushNamed(context, '/identify-landmark');
                  },
                  icon: Icon(Icons.camera_alt, color: Colors.white,),
                  label: Text('Identify Landmark', style: TextStyle(color: Colors.white),),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.purple,
                    padding: const EdgeInsets.symmetric(
                        horizontal: 20, vertical: 10),
                    textStyle: TextStyle(
                      fontSize: 18,
                        color: Colors.white
                    ),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(10),
                    ),
                  ),
                ),
                const SizedBox(height: 20),

                // Get Directions Button
                ElevatedButton.icon(
                  onPressed: () {
                    // TODO: Implement Get Directions feature
                  },
                  icon: Icon(Icons.navigation, color: Colors.white,),
                  label: Text('Get Directions', style: TextStyle(color: Colors.white),),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.purple,
                    padding: const EdgeInsets.symmetric(
                        horizontal: 20, vertical: 10),
                    textStyle: TextStyle(
                      fontSize: 18,
                      color: Colors.white
                    ),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(10),
                    ),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
