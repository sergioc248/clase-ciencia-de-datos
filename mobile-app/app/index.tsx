import { useState, useRef } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  ScrollView,
  ActivityIndicator,
  Alert,
  StyleSheet,
  Image,
} from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';
import * as ImagePicker from 'expo-image-picker';
import { Ionicons } from '@expo/vector-icons';
import { API_URL } from '../constants/config';

// Shape of each plate returned by the FastAPI /detect endpoint
interface PlateResult {
  text: string;
  confidence: number;
}

export default function HomeScreen() {
  const [permission, requestPermission] = useCameraPermissions();
  const cameraRef = useRef<CameraView>(null);

  const [capturedUri, setCapturedUri] = useState<string | null>(null);
  const [results, setResults] = useState<PlateResult[]>([]);
  const [loading, setLoading] = useState(false);

  // ---------- helpers ----------

  const sendToBackend = async (uri: string) => {
    setLoading(true);
    setResults([]);
    try {
      const formData = new FormData();
      // React Native's FormData accepts this object shape for files
      formData.append('file', { uri, name: 'plate.jpg', type: 'image/jpeg' } as unknown as Blob);

      const response = await fetch(`${API_URL}/detect`, {
        method: 'POST',
        body: formData,
        // Do NOT set Content-Type manually; fetch sets it with the boundary automatically
      });

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }

      const data = await response.json();
      // Expected response shape: { plates: PlateResult[] }
      setResults(data.plates ?? []);
    } catch (err) {
      Alert.alert('Error', `No se pudo conectar al servidor.\n${(err as Error).message}`);
    } finally {
      setLoading(false);
    }
  };

  // ---------- actions ----------

  const handleTakePhoto = async () => {
    if (!permission?.granted) {
      const { granted } = await requestPermission();
      if (!granted) {
        Alert.alert('Permiso denegado', 'Se necesita acceso a la cámara para tomar fotos.');
        return;
      }
    }

    const photo = await cameraRef.current?.takePictureAsync({ quality: 0.7 });
    if (photo?.uri) {
      setCapturedUri(photo.uri);
      await sendToBackend(photo.uri);
    }
  };

  const handlePickFromGallery = async () => {
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: 'images',
      quality: 0.7,
    });

    if (!result.canceled && result.assets.length > 0) {
      const uri = result.assets[0].uri;
      setCapturedUri(uri);
      await sendToBackend(uri);
    }
  };

  // ---------- permission gate ----------

  // Show a permission-request screen only when the status is known and denied
  if (permission && !permission.granted && !permission.canAskAgain) {
    return (
      <View style={styles.centered}>
        <Text style={styles.permissionText}>
          Acceso a la cámara denegado. Habilítalo en Configuración para tomar fotos.
        </Text>
      </View>
    );
  }

  // ---------- render ----------

  return (
    <View style={styles.container}>
      {/* Camera preview – always visible as the main viewfinder */}
      <View style={styles.cameraWrapper}>
        {permission?.granted ? (
          <CameraView ref={cameraRef} style={styles.camera} facing="back" />
        ) : (
          <View style={[styles.camera, styles.cameraPlaceholder]}>
            <Ionicons name="camera-outline" size={64} color="#555" />
            <Text style={styles.placeholderText}>Cámara no disponible</Text>
          </View>
        )}

        {/* Show thumbnail of last captured/picked image in the corner */}
        {capturedUri && (
          <Image source={{ uri: capturedUri }} style={styles.thumbnail} />
        )}
      </View>

      {/* Action buttons */}
      <View style={styles.buttonRow}>
        <TouchableOpacity
          style={styles.actionButton}
          onPress={handleTakePhoto}
          disabled={loading}
        >
          <Ionicons name="camera" size={28} color="#fff" />
          <Text style={styles.buttonLabel}>Tomar foto</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={styles.actionButton}
          onPress={handlePickFromGallery}
          disabled={loading}
        >
          <Ionicons name="images" size={28} color="#fff" />
          <Text style={styles.buttonLabel}>Galería</Text>
        </TouchableOpacity>
      </View>

      {/* Results area */}
      <ScrollView style={styles.resultsContainer} contentContainerStyle={styles.resultsContent}>
        {loading && (
          <View style={styles.loadingWrapper}>
            <ActivityIndicator size="large" color="#e94560" />
            <Text style={styles.loadingText}>Detectando placas…</Text>
          </View>
        )}

        {!loading && results.length === 0 && capturedUri && (
          <Text style={styles.emptyText}>No se detectaron placas.</Text>
        )}

        {!loading && results.length === 0 && !capturedUri && (
          <Text style={styles.emptyText}>Toma una foto o selecciona una imagen para comenzar.</Text>
        )}

        {!loading &&
          results.map((plate, index) => (
            <View key={index} style={styles.plateCard}>
              {/* Plate text in large monospace */}
              <Text style={styles.plateText}>{plate.text}</Text>
              {/* Confidence as percentage */}
              <Text style={styles.confidenceText}>
                Confianza: {(plate.confidence * 100).toFixed(1)}%
              </Text>
            </View>
          ))}
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#1a1a2e',
  },
  centered: {
    flex: 1,
    backgroundColor: '#1a1a2e',
    justifyContent: 'center',
    alignItems: 'center',
    padding: 24,
  },
  permissionText: {
    color: '#ccc',
    textAlign: 'center',
    fontSize: 16,
  },

  // Camera
  cameraWrapper: {
    flex: 1,
    position: 'relative',
  },
  camera: {
    flex: 1,
  },
  cameraPlaceholder: {
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#0d0d1a',
  },
  placeholderText: {
    color: '#555',
    marginTop: 8,
    fontSize: 14,
  },
  thumbnail: {
    position: 'absolute',
    bottom: 8,
    right: 8,
    width: 72,
    height: 72,
    borderRadius: 8,
    borderWidth: 2,
    borderColor: '#e94560',
  },

  // Buttons
  buttonRow: {
    flexDirection: 'row',
    justifyContent: 'space-evenly',
    paddingVertical: 16,
    paddingHorizontal: 24,
    backgroundColor: '#16213e',
  },
  actionButton: {
    alignItems: 'center',
    gap: 6,
    paddingVertical: 10,
    paddingHorizontal: 28,
    backgroundColor: '#e94560',
    borderRadius: 12,
  },
  buttonLabel: {
    color: '#fff',
    fontSize: 13,
    fontWeight: '600',
  },

  // Results
  resultsContainer: {
    maxHeight: 280,
    backgroundColor: '#1a1a2e',
  },
  resultsContent: {
    padding: 16,
    gap: 12,
  },
  loadingWrapper: {
    alignItems: 'center',
    paddingVertical: 24,
    gap: 12,
  },
  loadingText: {
    color: '#ccc',
    fontSize: 14,
  },
  emptyText: {
    color: '#666',
    textAlign: 'center',
    fontSize: 14,
    paddingVertical: 16,
  },
  plateCard: {
    backgroundColor: '#16213e',
    borderRadius: 12,
    padding: 16,
    borderLeftWidth: 4,
    borderLeftColor: '#e94560',
  },
  plateText: {
    fontFamily: 'monospace', // monospace for license plate feel
    fontSize: 28,
    fontWeight: 'bold',
    color: '#ffffff',
    letterSpacing: 4,
  },
  confidenceText: {
    marginTop: 6,
    fontSize: 13,
    color: '#888',
  },
});
