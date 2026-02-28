import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

// ─── State ───────────────────────────────────────────────────────────────────
let metadata = null;
let isWorldSpace = false;  // true when coord_space === 'world'
let verticesData = null;   // Float32Array (T * V * 3) — unused in rigged mode
let facesData = null;      // Int32Array (F * 3) — unused in rigged mode

// Rigged model state
let riggedRoot = null;        // gltf.scene root added to scene (for removal on swap)
let riggedModel = null;       // THREE.SkinnedMesh from GLB
let riggedSkeleton = null;    // THREE.Skeleton
let riggedBones = {};         // {name: THREE.Bone} lookup
let boneRotationsData = null; // Float32Array (T * N_BONES * 4) — quaternion xyzw
let rootPositionData = null;  // Float32Array (T * 3)
let isRigged = false;         // true when rigged model is loaded
let modelScale = 1.0;         // uniform scale to match athlete size
let groundOffset = 0;          // Y offset to align mannequin feet with ground plane
let handBoneLength = 0;       // GLB local units: wrist to knuckle
const handRestPositions = {};  // {name: number} — rest Y position for hand bones
let keypointsData = null;  // Float32Array (T * 70 * 3)
let hammerData = null;     // Float32Array (T * 3)
let simHammerFwdData = null;  // Float32Array (T * 3) — forward sim ball
let simHammerBwdData = null;  // Float32Array (T * 3) — backward sim ball
let simGripFwdData = null;   // Float32Array (T * 3) — forward sim grip point
let simGripBwdData = null;   // Float32Array (T * 3) — backward sim grip point
let fillTypeData = null;   // Int8Array (T) — provenance per frame
let hammerProvenanceData = null; // Int8Array (T) — 0=LIFTED, 1=FALLBACK, 2=SCREENED, 3=NAN

let currentFrame = 0;
let playing = false;
let lastFrameTime = 0;
let playbackSpeed = 1.0;
let timelineMin = 0;
let timelineMax = 0;

// Three.js objects
let renderer, scene, camera, controls;
let bodyMesh, hammerSphere, hammerWireLine, simFwdSphere, simFwdWireLine, simBwdSphere, simBwdWireLine, groundGroup;
let leftLegPlane, rightLegPlane, legPlanesGroup;
let circleGroup, circleLine;
let circleLabelsGroup = null;
let labelZeroMesh = null;
let isDraggingLabel = false;
let labelDragStartAngle = 0;
let labelDragStartRotation = 0;
const labelRaycaster = new THREE.Raycaster();
const labelGroundPlane = new THREE.Plane();
let backPlanesGroup, backPlane;
let circlePositionsData = null;  // Float32Array (T * 3) — per-frame circle center (camera space)
let poleGroup;  // kept for legacy data compat — no longer rendered
let groundY = -Infinity;

// Separation metric
let separationAngles = null;   // Float32Array(T)
let separationEnabled = true;
let torsoMask = null;          // Uint8Array(V) — 1 for torso vertices (rigged mode)
const BASE_R = 0x77/255, BASE_G = 0x99/255, BASE_B = 0xBB/255;  // rigged mannequin base color

// Pipeline-precomputed metric arrays (loaded from binary when available)
let pipelineSeparation = null;  // Float32Array(T) — sep_pelvis_thorax_deg from analytics
let pipelineBackLean = null;    // Float32Array(T) — back_lean_deg from analytics

// Support state
let supportStateData = null;   // Int8Array(T) — 1=SS, 2=DS (matches analytics pipeline)

// Orbit extremes (high/low points per turn)
let orbitExtremes = [];        // [{frame, pos:[x,y,z], type:'high'|'low', turnFrameCount, sphere}, ...]
let orbitExtremesGroup = null;

// Per-vertex colors (from paint_mesh_from_video.py)
let vertexColorsData = null;   // Float32Array(V * 3) — static RGB per vertex

// Wire distance + 2D reprojection overlay
let wireDistanceData = null;   // Float32Array(T) — wire length per frame
let hammerReproj2d = null;     // Float32Array(T*2) — reprojected 3D hammer in px
let hammerOrig2d = null;       // Float32Array(T*2) — original 2D detection in px

// PiP preloaded thumbnails
let pipFrames = null;          // Array of Image objects (preloaded)

// MHR70 keypoint indices
const KP_LEFT_HIP = 9, KP_RIGHT_HIP = 10;
const KP_LEFT_KNEE = 11, KP_RIGHT_KNEE = 12;
const KP_LEFT_ANKLE = 13, KP_RIGHT_ANKLE = 14;
const KP_LEFT_SHOULDER = 5, KP_RIGHT_SHOULDER = 6;
const PLANE_WIDTH = 0.35;   // meters
const PLANE_EXT = 0.30;     // ~1 foot past ankle

// SAM3D camera space: X-right, Y-down, Z-forward
// Three.js: X-right, Y-up, Z-backward
// Flip: (x, -y, -z)  — only for camera-space data
function camToThree(x, y, z) {
  if (isWorldSpace) return [x, y, z];
  return [x, -y, -z];
}

// ─── Keypoint Accessor ──────────────────────────────────────────────────────

function getKp(frame, idx) {
  const off = (frame * 70 + idx) * 3;
  if (isWorldSpace) {
    return new THREE.Vector3(keypointsData[off], keypointsData[off + 1], keypointsData[off + 2]);
  }
  return new THREE.Vector3(keypointsData[off], -keypointsData[off + 1], -keypointsData[off + 2]);
}

// ─── Grid Texture (one cell, tiles via repeat) ─────────────────────────────

function createPlaneTexture(r, g, b) {
  const s = 64;
  const c = document.createElement('canvas');
  c.width = s; c.height = s;
  const ctx = c.getContext('2d');

  // Solid fill
  ctx.fillStyle = `rgb(${r},${g},${b})`;
  ctx.fillRect(0, 0, s, s);

  // Grid lines at left + top edges (tiling creates full grid)
  ctx.strokeStyle = 'rgba(255,255,255,0.35)';
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(0.5, 0); ctx.lineTo(0.5, s); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(0, 0.5); ctx.lineTo(s, 0.5); ctx.stroke();

  const tex = new THREE.CanvasTexture(c);
  tex.wrapS = THREE.RepeatWrapping;
  tex.wrapT = THREE.RepeatWrapping;
  return tex;
}

// ─── Data Loading ────────────────────────────────────────────────────────────

// Loading progress tracking
let totalBytes = 0;
let loadedBytes = 0;

function updateLoadingProgress() {
  const bar = document.getElementById('loading-bar');
  const bytesLabel = document.getElementById('loading-bytes');
  if (bar && totalBytes > 0) {
    bar.style.width = Math.min(100, (loadedBytes / totalBytes) * 100).toFixed(1) + '%';
  }
  if (bytesLabel && totalBytes > 0) {
    bytesLabel.textContent = `${(loadedBytes / 1048576).toFixed(1)} / ${(totalBytes / 1048576).toFixed(1)} MB`;
  }
}

async function loadBinary(url, dtype) {
  const cacheBust = url + (url.includes('?') ? '&' : '?') + '_t=' + Date.now();
  const resp = await fetch(cacheBust);
  const contentLength = parseInt(resp.headers.get('Content-Length'), 10);
  if (contentLength) totalBytes += contentLength;
  updateLoadingProgress();

  // Stream the response to track progress
  if (resp.body && contentLength > 100000) {
    const reader = resp.body.getReader();
    const chunks = [];
    let received = 0;
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      received += value.length;
      loadedBytes += value.length;
      updateLoadingProgress();
    }
    const buf = new Uint8Array(received);
    let offset = 0;
    for (const chunk of chunks) {
      buf.set(chunk, offset);
      offset += chunk.length;
    }
    if (dtype === 'float32') return new Float32Array(buf.buffer);
    if (dtype === 'int32') return new Int32Array(buf.buffer);
    if (dtype === 'int8') return new Int8Array(buf.buffer);
    return buf;
  }

  // Small files — no progress tracking needed
  const buf = await resp.arrayBuffer();
  if (contentLength) loadedBytes += contentLength;
  updateLoadingProgress();
  if (dtype === 'float32') return new Float32Array(buf);
  if (dtype === 'int32') return new Int32Array(buf);
  if (dtype === 'int8') return new Int8Array(buf);
  return new Uint8Array(buf);
}

async function loadData() {
  const resp = await fetch('metadata.json');
  metadata = await resp.json();
  isWorldSpace = metadata.coord_space === 'world';

  isRigged = !!metadata.rigged;

  if (isRigged) {
    // Rigged mode: load keypoints + hammer + bone data (no vertices/faces)
    const [kps, hammer, boneRots, rootPos] = await Promise.all([
      loadBinary(metadata.files.keypoints, 'float32'),
      loadBinary(metadata.files.hammer, 'float32'),
      loadBinary(metadata.files.bone_rotations, 'float32'),
      loadBinary(metadata.files.root_position, 'float32'),
    ]);
    keypointsData = kps;
    hammerData = hammer;
    boneRotationsData = boneRots;
    rootPositionData = rootPos;
    console.log(`Rigged mode: ${metadata.bone_count} bones, ${metadata.frame_count} frames`);
  } else {
    // Legacy mode: load vertices + faces
    const [verts, faces, kps, hammer] = await Promise.all([
      loadBinary(metadata.files.vertices, 'float32'),
      loadBinary(metadata.files.faces, 'int32'),
      loadBinary(metadata.files.keypoints, 'float32'),
      loadBinary(metadata.files.hammer, 'float32'),
    ]);
    verticesData = verts;
    facesData = faces;
    keypointsData = kps;
    hammerData = hammer;
  }

  // Load fill_type if available
  if (metadata.files.fill_type) {
    fillTypeData = await loadBinary(metadata.files.fill_type, 'int8');
  }

  // Load per-frame circle positions if available
  if (metadata.files.circle_positions) {
    circlePositionsData = await loadBinary(metadata.files.circle_positions, 'float32');
  }

  // Load support state (SS/DS) if available
  if (metadata.files.support_state) {
    supportStateData = await loadBinary(metadata.files.support_state, 'int8');
  }

  // Load leg alignment (precomputed from analytics) if available
  if (metadata.files.leg_alignment) {
    legAlignmentData = await loadBinary(metadata.files.leg_alignment, 'float32');
  }

  // Load precomputed separation and back lean from analytics pipeline
  if (metadata.files.separation) {
    pipelineSeparation = await loadBinary(metadata.files.separation, 'float32');
    console.log(`Loaded pipeline separation: ${pipelineSeparation.length} frames`);
  }
  if (metadata.files.back_lean) {
    pipelineBackLean = await loadBinary(metadata.files.back_lean, 'float32');
    console.log(`Loaded pipeline back lean: ${pipelineBackLean.length} frames`);
  }

  // Load static vertex colors if available (from paint_mesh_from_video.py)
  if (metadata.files.vertex_colors) {
    vertexColorsData = await loadBinary(metadata.files.vertex_colors, 'float32');
  }

  // Load hammer provenance if available
  if (metadata.files.hammer_provenance) {
    hammerProvenanceData = await loadBinary(metadata.files.hammer_provenance, 'int8');
    console.log(`Loaded hammer provenance: ${hammerProvenanceData.length} frames`);
  }

  // Load sim hammer overlays (forward + backward) if available
  if (metadata.files.sim_hammer_fwd) {
    simHammerFwdData = await loadBinary(metadata.files.sim_hammer_fwd, 'float32');
    console.log(`Loaded sim hammer [fwd]: ${simHammerFwdData.length / 3} frames`);
  } else if (metadata.files.sim_hammer) {
    // Legacy single-file fallback
    simHammerFwdData = await loadBinary(metadata.files.sim_hammer, 'float32');
    console.log(`Loaded sim hammer: ${simHammerFwdData.length / 3} frames`);
  }
  if (metadata.files.sim_hammer_bwd) {
    simHammerBwdData = await loadBinary(metadata.files.sim_hammer_bwd, 'float32');
    console.log(`Loaded sim hammer [bwd]: ${simHammerBwdData.length / 3} frames`);
  }
  // Load sim grip points (actual wire attachment used by sim)
  if (metadata.files.sim_grip_fwd) {
    simGripFwdData = await loadBinary(metadata.files.sim_grip_fwd, 'float32');
    console.log(`Loaded sim grip [fwd]: ${simGripFwdData.length / 3} frames`);
  }
  if (metadata.files.sim_grip_bwd) {
    simGripBwdData = await loadBinary(metadata.files.sim_grip_bwd, 'float32');
    console.log(`Loaded sim grip [bwd]: ${simGripBwdData.length / 3} frames`);
  }

  // Load wire distance + 2D reprojection data
  if (metadata.files.wire_distance)
    wireDistanceData = await loadBinary(metadata.files.wire_distance, 'float32');
  if (metadata.files.hammer_reproj_2d)
    hammerReproj2d = await loadBinary(metadata.files.hammer_reproj_2d, 'float32');
  if (metadata.files.hammer_orig_2d)
    hammerOrig2d = await loadBinary(metadata.files.hammer_orig_2d, 'float32');

  // Preload PiP thumbnail frames
  if (metadata.pip_thumbnails) {
    const thumbDir = metadata.pip_thumbnails.dir;
    const T = metadata.frame_count;
    pipFrames = new Array(T);
    const loadPromises = [];
    for (let f = 0; f < T; f++) {
      const padded = String(f).padStart(5, '0');
      const img = new window.Image();
      const promise = new Promise((resolve) => {
        img.onload = resolve;
        img.onerror = resolve;  // don't block on missing frames
      });
      img.src = `${thumbDir}/frame_${padded}.jpg`;
      pipFrames[f] = img;
      loadPromises.push(promise);
    }
    await Promise.all(loadPromises);
  } else {
    // Fallback: try to preload from ../frames/ (full-res, legacy path)
    const T = metadata.frame_count;
    pipFrames = new Array(T);
    const loadPromises = [];
    for (let f = 0; f < T; f++) {
      const padded = String(f).padStart(5, '0');
      const img = new window.Image();
      const promise = new Promise((resolve) => {
        img.onload = resolve;
        img.onerror = resolve;
      });
      img.src = `../frames/frame_${padded}.jpg`;
      pipFrames[f] = img;
      loadPromises.push(promise);
    }
    await Promise.all(loadPromises);
  }
}

// ─── Scene Setup ─────────────────────────────────────────────────────────────

function initScene() {
  const container = document.getElementById('canvas-container');

  // Renderer
  renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setClearColor(0x1a1a1a);
  container.appendChild(renderer.domElement);

  // Scene — no rotation needed, coordinate flip applied per-vertex
  scene = new THREE.Scene();

  // Camera
  if (isWorldSpace) {
    // World space: 60° FOV, position/target from metadata
    camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 200);
    const cp = metadata.camera_position;
    const ct = metadata.camera_target;
    const cu = metadata.camera_up;
    camera.position.set(cp[0], cp[1], cp[2]);
    camera.up.set(cu[0], cu[1], cu[2]);
    camera.lookAt(ct[0], ct[1], ct[2]);
  } else {
    // Camera space: match SAM3D focal length
    const imageHeight = 1080;
    const focalLength = metadata.focal_length || 2200;
    const fovDeg = 2 * Math.atan(imageHeight / (2 * focalLength)) * (180 / Math.PI);
    camera = new THREE.PerspectiveCamera(fovDeg, window.innerWidth / window.innerHeight, 0.1, 100);
    camera.position.set(0, 0, 0);
  }

  // Controls — orbit around the body center
  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.1;

  // Lights
  const ambient = new THREE.AmbientLight(0xffffff, 0.6);
  scene.add(ambient);
  const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
  dirLight.position.set(2, 5, 5);
  scene.add(dirLight);

  // Resize handler
  window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
    applyTimelineRange();
    // Re-size overlay canvas if graph is active
    if (graphOverlayActive) {
      const overlayId = activeOverlaySource === 'separation' ? 'separation-container' : 'kneeangle-container';
      const drawFn = activeOverlaySource === 'separation' ? drawSeparationGraph : drawLegCorotationGraph;
      const oc = document.getElementById(overlayId);
      if (oc && oc.style.display !== 'none') {
        const canvas = oc.querySelector('canvas');
        if (canvas) {
          requestAnimationFrame(() => {
            const rect = canvas.getBoundingClientRect();
            canvas.width = Math.round(rect.width);
            canvas.height = Math.round(rect.height);
            drawFn(currentFrame);
          });
        }
      }
    }
  });
}

// ─── Mesh ────────────────────────────────────────────────────────────────────

function createBodyMesh() {
  const V = metadata.vertex_count;
  const F = metadata.face_count;

  const geometry = new THREE.BufferGeometry();

  // Position attribute — pre-allocate, will be updated each frame
  const posArray = new Float32Array(V * 3);
  const posAttr = new THREE.BufferAttribute(posArray, 3);
  posAttr.setUsage(THREE.StreamDrawUsage);
  geometry.setAttribute('position', posAttr);

  // Vertex colors — use painted colors if available, else blue
  const colorArray = new Float32Array(V * 3);
  if (vertexColorsData && vertexColorsData.length === V * 3) {
    colorArray.set(vertexColorsData);
  } else {
    for (let i = 0; i < V; i++) {
      colorArray[i * 3]     = SKIN_R;
      colorArray[i * 3 + 1] = SKIN_G;
      colorArray[i * 3 + 2] = SKIN_B;
    }
  }
  const colorAttr = new THREE.BufferAttribute(colorArray, 3);
  colorAttr.setUsage(THREE.StreamDrawUsage);
  geometry.setAttribute('color', colorAttr);

  // Index (faces) — set once
  const indexArray = new Uint32Array(F * 3);
  for (let i = 0; i < F * 3; i++) {
    indexArray[i] = facesData[i];
  }
  geometry.setIndex(new THREE.BufferAttribute(indexArray, 1));

  const material = new THREE.MeshStandardMaterial({
    color: 0xffffff,
    vertexColors: true,
    roughness: 0.6,
    metalness: 0.05,
    side: THREE.DoubleSide,
  });

  bodyMesh = new THREE.Mesh(geometry, material);
  scene.add(bodyMesh);
}

function updateMeshFrame(frame) {
  const V = metadata.vertex_count;
  const offset = frame * V * 3;
  const posAttr = bodyMesh.geometry.getAttribute('position');
  const arr = posAttr.array;

  if (isWorldSpace) {
    // World space: data already in Y-up, copy directly
    arr.set(verticesData.subarray(offset, offset + V * 3));
  } else {
    // Camera space: apply SAM3D→Three.js coordinate flip
    for (let i = 0; i < V; i++) {
      const si = offset + i * 3;
      const di = i * 3;
      arr[di]     =  verticesData[si];      // X unchanged
      arr[di + 1] = -verticesData[si + 1];  // Y flipped
      arr[di + 2] = -verticesData[si + 2];  // Z flipped
    }
  }

  posAttr.needsUpdate = true;
  bodyMesh.geometry.computeVertexNormals();
}

// ─── Rigged Model ─────────────────────────────────────────────────────────────

async function loadRiggedModel(overrideModelUrl) {
  const modelUrl = (overrideModelUrl || metadata.model || 'models/humanoid.glb') + '?v=' + Date.now();
  const loader = new GLTFLoader();

  // Clean up previous rigged model if swapping
  if (riggedRoot) {
    scene.remove(riggedRoot);
    riggedRoot = null;
    riggedModel = null;
    riggedSkeleton = null;
    riggedBones = {};
    torsoMask = null;
  }

  return new Promise((resolve, reject) => {
    loader.load(modelUrl, (gltf) => {
      const root = gltf.scene;

      // Find the SkinnedMesh
      root.traverse((child) => {
        if (child.isSkinnedMesh) {
          riggedModel = child;
          riggedSkeleton = child.skeleton;
        }
      });

      if (!riggedModel) {
        console.error('No SkinnedMesh found in GLB');
        reject(new Error('No SkinnedMesh in GLB'));
        return;
      }

      // Build bone name → bone lookup
      riggedSkeleton.bones.forEach((bone) => {
        riggedBones[bone.name] = bone;
      });

      console.log('Rigged model loaded:', Object.keys(riggedBones).join(', '));

      // Save rest positions for hand bones (before any frame updates)
      for (const hn of ['LeftHand', 'RightHand']) {
        if (riggedBones[hn]) handRestPositions[hn] = riggedBones[hn].position.clone();
      }

      initFingerCurl();

      // Uniform scale to match athlete height
      modelScale = metadata.model_scale || 1.0;
      console.log(`Model scale: ${modelScale.toFixed(4)}`);
      root.scale.set(modelScale, modelScale, modelScale);

      // Build torso vertex mask from skinning data
      const torsoBoneNames = new Set(['Hips', 'Spine', 'Chest', 'UpperChest']);
      const skinIndexAttr = riggedModel.geometry.getAttribute('skinIndex');
      const skinWeightAttr = riggedModel.geometry.getAttribute('skinWeight');
      const vertCount = riggedModel.geometry.getAttribute('position').count;
      const boneIndexToName = {};
      riggedSkeleton.bones.forEach((bone, idx) => {
        boneIndexToName[idx] = bone.name;
      });
      torsoMask = new Uint8Array(vertCount);
      let torsoVertCount = 0;
      for (let v = 0; v < vertCount; v++) {
        let torsoWeight = 0;
        for (let j = 0; j < 4; j++) {
          const boneIdx = skinIndexAttr.getComponent(v, j);
          const weight = skinWeightAttr.getComponent(v, j);
          if (torsoBoneNames.has(boneIndexToName[boneIdx])) {
            torsoWeight += weight;
          }
        }
        if (torsoWeight > 0.5) { torsoMask[v] = 1; torsoVertCount++; }
      }
      console.log(`Torso mask: ${torsoVertCount}/${vertCount} vertices`);

      // Add vertex color attribute (initialize to base color)
      const colorArray = new Float32Array(vertCount * 3);
      for (let i = 0; i < vertCount; i++) {
        colorArray[i * 3]     = BASE_R;
        colorArray[i * 3 + 1] = BASE_G;
        colorArray[i * 3 + 2] = BASE_B;
      }
      riggedModel.geometry.setAttribute('color', new THREE.BufferAttribute(colorArray, 3));

      // Material with vertex colors
      riggedModel.material = new THREE.MeshStandardMaterial({
        vertexColors: true,
        roughness: 0.6,
        metalness: 0.05,
        side: THREE.DoubleSide,
        transparent: true,
        opacity: 0.85,
      });

      riggedRoot = root;
      scene.add(root);

      // Compute groundOffset from actual foot position:
      // 1. Position at frame 0 with groundOffset=0
      // 2. Measure foot world Y
      // 3. Offset to ground_y
      groundOffset = 0;
      if (rootPositionData) {
        updateRiggedFrame(0);
        root.updateMatrixWorld(true);

        // Find lowest foot bone world position
        const footBoneNames = ['LeftToe_End', 'RightToe_End', 'LeftToes', 'RightToes'];
        let lowestY = Infinity;
        const tmpPos = new THREE.Vector3();
        for (const fn of footBoneNames) {
          const fb = riggedBones[fn];
          if (fb) {
            fb.getWorldPosition(tmpPos);
            if (tmpPos.y < lowestY) lowestY = tmpPos.y;
          }
        }
        if (lowestY < Infinity) {
          groundOffset = (metadata.ground_y || 0) - lowestY;
          console.log(`Ground offset: ${groundOffset.toFixed(4)}m (foot=${lowestY.toFixed(4)}, ground=${metadata.ground_y})`);
          updateRiggedFrame(0);  // re-position with correct groundOffset
        }
      }

      resolve();
    }, undefined, (err) => {
      console.error('GLB load error:', err);
      reject(err);
    });
  });
}

// Pistol grip finger quaternions extracted from Mixamo "Pistol Idle" animation.
// These are pose-bone local rotations (delta from rest). Format: [x, y, z, w] (Three.js order).
const PISTOL_GRIP = {
  "L_Thumb_1": [-0.151985, -0.095355, -0.034586, 0.983164],
  "L_Thumb_2": [-0.094063, 0.070201, 0.031516, 0.992588],
  "L_Thumb_3": [0.02718, 0.017916, 0.218367, 0.975323],
  "L_Index_1": [0.271488, -0.012903, -0.019737, 0.962153],
  "L_Index_2": [0.512289, -0.0, -0.052502, 0.857207],
  "L_Index_3": [0.207034, 0.0, -0.021218, 0.978104],
  "L_Middle_1": [0.354152, -0.025176, -0.006605, 0.934826],
  "L_Middle_2": [0.500314, -0.0, -0.051275, 0.864325],
  "L_Middle_3": [0.33513, -0.0, -0.034346, 0.941546],
  "L_Ring_1": [0.420829, -0.047302, 0.023515, 0.905601],
  "L_Ring_2": [0.508183, 7.8e-05, -0.05195, 0.859681],
  "L_Ring_3": [0.173008, -1.4e-05, -0.01781, 0.984759],
  "L_Little_1": [0.324574, -0.049477, 0.07739, 0.94139],
  "L_Little_2": [0.49149, 0.0, -0.05037, 0.869425],
  "L_Little_3": [0.273045, -0.0, -0.027983, 0.961594],
  "R_Thumb_1": [0.098292, 0.010443, 0.077123, 0.99211],
  "R_Thumb_2": [-0.111955, -0.187873, 0.210885, 0.952731],
  "R_Thumb_3": [-0.272084, -0.018267, 0.407849, 0.871376],
  "R_Index_1": [0.21981, 0.00371, 0.038577, 0.974773],
  "R_Index_2": [0.420375, -0.0, 0.03425, 0.906704],
  "R_Index_3": [0.431044, 2e-06, 0.035122, 0.901647],
  "R_Middle_1": [0.430214, -0.035448, 0.142209, 0.89075],
  "R_Middle_2": [0.825171, 0.0, 0.06723, 0.560868],
  "R_Middle_3": [0.162822, -2e-06, 0.013256, 0.986566],
  "R_Ring_1": [0.478968, 0.00177, 0.070173, 0.875021],
  "R_Ring_2": [0.753735, -0.0, 0.06141, 0.654303],
  "R_Ring_3": [0.417158, -5e-06, 0.033977, 0.908199],
  "R_Little_1": [0.491661, 0.018615, 0.041323, 0.869607],
  "R_Little_2": [0.583947, 0.0, 0.047577, 0.810396],
  "R_Little_3": [0.435195, -2e-06, 0.035453, 0.899638],
};

const fingerRestQuats = {};
const fingerCurledQuats = {};

function initFingerCurl() {
  for (const [name, xyzw] of Object.entries(PISTOL_GRIP)) {
    const bone = riggedBones[name];
    if (!bone) continue;
    fingerRestQuats[name] = bone.quaternion.clone();
    const delta = new THREE.Quaternion(xyzw[0], xyzw[1], xyzw[2], xyzw[3]);
    fingerCurledQuats[name] = bone.quaternion.clone().multiply(delta);
  }
}

function updateRiggedFrame(frame) {
  if (!riggedSkeleton || !boneRotationsData || !rootPositionData) return;

  const nBones = metadata.bone_count;
  const boneNames = metadata.bone_names;

  // Set root position (Hips) — world coords divided by modelScale (root has uniform scale)
  // groundOffset aligns feet with ground plane
  const rOff = frame * 3;
  const hipsBone = riggedBones['Hips'];
  if (hipsBone && rootPositionData) {
    hipsBone.position.set(
      rootPositionData[rOff] / modelScale,
      (rootPositionData[rOff + 1] + groundOffset) / modelScale,
      rootPositionData[rOff + 2] / modelScale
    );
  }

  // Set bone quaternions
  for (let i = 0; i < nBones; i++) {
    const bone = riggedBones[boneNames[i]];
    if (!bone) continue;

    const qOff = (frame * nBones + i) * 4;
    const qx = boneRotationsData[qOff];
    const qy = boneRotationsData[qOff + 1];
    const qz = boneRotationsData[qOff + 2];
    const qw = boneRotationsData[qOff + 3];

    bone.quaternion.set(qx, qy, qz, qw);
  }

  // Grip pose: rotate forearms toward grip point so arms converge smoothly (pre-release only)
  const releaseFrame = metadata.throw_window && metadata.throw_window.release;
  const isPreRelease = releaseFrame == null || frame <= releaseFrame;
  if (isPreRelease && keypointsData) {
    const L_WRIST = 62, R_WRIST = 41;
    const lWrist = getKp(frame, L_WRIST);
    const rWrist = getKp(frame, R_WRIST);
    const wristMid = lWrist.clone().add(rWrist).multiplyScalar(0.5);

    // Aim point: extend past wrist midpoint by hand_length + 1 inch
    // so the arm-line intersection falls 1" beyond the knuckles
    const hips = riggedBones['Hips'];
    const hipsWorld = new THREE.Vector3();
    if (hips) hips.getWorldPosition(hipsWorld);
    const outDir = wristMid.clone().sub(hipsWorld).normalize();
    const HAND_LEN = 0.08;   // ~8cm wrist to knuckles
    const ONE_INCH = 0.0254;
    const gripPt = wristMid.clone().add(outDir.multiplyScalar((HAND_LEN + ONE_INCH) / modelScale * 0.5));

    for (const forearmName of ['LeftForeArm', 'RightForeArm']) {
      const forearm = riggedBones[forearmName];
      const handName = forearmName.replace('ForeArm', 'Hand');
      const hand = riggedBones[handName];
      if (!forearm || !hand) continue;

      // Current direction: forearm → hand (where keypoints placed it)
      forearm.parent.updateWorldMatrix(true, false);
      forearm.updateWorldMatrix(true, false);
      const forearmWorld = new THREE.Vector3();
      const handWorld = new THREE.Vector3();
      forearm.getWorldPosition(forearmWorld);
      hand.getWorldPosition(handWorld);
      const currentDir = handWorld.clone().sub(forearmWorld).normalize();

      // Desired direction: forearm → grip point
      const desiredDir = gripPt.clone().sub(forearmWorld).normalize();

      // Delta rotation in world space
      const deltaWorld = new THREE.Quaternion().setFromUnitVectors(currentDir, desiredDir);

      // Convert to local: localDelta = parentWorldInv * deltaWorld * parentWorld
      // Then apply: forearm.quaternion = localDelta * forearm.quaternion
      const parentWorldQ = new THREE.Quaternion();
      forearm.parent.getWorldQuaternion(parentWorldQ);
      const parentInv = parentWorldQ.clone().invert();
      const deltaLocal = parentInv.clone().multiply(deltaWorld).multiply(parentWorldQ);
      forearm.quaternion.premultiply(deltaLocal);

      // Roll hand 90° around forearm axis so palms face each other (thumb up, pinky down)
      forearm.updateWorldMatrix(true, false);
      hand.updateWorldMatrix(true, false);
      const fwPost = new THREE.Vector3(); forearm.getWorldPosition(fwPost);
      const hwPost = new THREE.Vector3(); hand.getWorldPosition(hwPost);
      const rollAxis = hwPost.clone().sub(fwPost).normalize();
      const rollAngle = (forearmName === 'LeftForeArm') ? -Math.PI / 2 : Math.PI / 2;
      const rollWorld = new THREE.Quaternion().setFromAxisAngle(rollAxis, rollAngle);
      const handParentQ = new THREE.Quaternion();
      hand.parent.getWorldQuaternion(handParentQ);
      const handParentInv = handParentQ.clone().invert();
      const rollLocal = handParentInv.clone().multiply(rollWorld).multiply(handParentQ);
      hand.quaternion.premultiply(rollLocal);
    }
  }
  const fingerQuats = isPreRelease ? fingerCurledQuats : fingerRestQuats;
  for (const [name, q] of Object.entries(fingerQuats)) {
    const bone = riggedBones[name];
    if (bone) bone.quaternion.copy(q);
  }
}

// ─── Hammer ──────────────────────────────────────────────────────────────────

// fill_type colors: 0=raw (orange), 1=manual (green), 2=propagated (cyan),
//                   3=kink_replaced (magenta), 4=spline (red)
const FILL_TYPE_COLORS = [
  { color: 0xff8c00, emissive: 0xff6600, label: 'Raw', css: '#ff8c00' },
  { color: 0x00cc66, emissive: 0x009944, label: 'Manual', css: '#00cc66' },
  { color: 0x00bbff, emissive: 0x0088cc, label: 'Propagated', css: '#00bbff' },
  { color: 0xdd44ff, emissive: 0xaa22cc, label: 'Kink Fix', css: '#dd44ff' },
  { color: 0xff3333, emissive: 0xcc1111, label: 'Spline', css: '#ff3333' },
];

function createHammer() {
  const isDiscus = metadata && metadata.event === 'discus';
  if (isDiscus) {
    // Flat disc: radius 0.11m, height 0.02m (WA standard discus ~220mm diameter)
    const geo = new THREE.CylinderGeometry(0.11, 0.11, 0.02, 32);
    const mat = new THREE.MeshStandardMaterial({
      color: 0xcc6600,
      emissive: 0x993300,
      emissiveIntensity: 0.3,
      roughness: 0.5,
      metalness: 0.3,
    });
    hammerSphere = new THREE.Mesh(geo, mat);
    scene.add(hammerSphere);
    // No wire for discus
    hammerWireLine = null;
  } else {
    const geo = new THREE.SphereGeometry(0.08, 16, 16);
    const mat = new THREE.MeshStandardMaterial({
      color: 0xff8c00,
      emissive: 0xff6600,
      emissiveIntensity: 0.3,
      roughness: 0.3,
      metalness: 0.4,
    });
    hammerSphere = new THREE.Mesh(geo, mat);
    scene.add(hammerSphere);

    // Wire line from wrist midpoint to tracked hammer ball
    const hamWireMat = new THREE.LineBasicMaterial({
      color: 0xff8c00, transparent: true, opacity: 0.5,
    });
    const hamWireGeo = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(), new THREE.Vector3(),
    ]);
    hammerWireLine = new THREE.Line(hamWireGeo, hamWireMat);
    scene.add(hammerWireLine);
  }

  // Sim hammer overlay — forward (green)
  if (simHammerFwdData) {
    const simGeo = new THREE.SphereGeometry(0.07, 16, 16);
    const simMat = new THREE.MeshStandardMaterial({
      color: 0x00ff88, emissive: 0x00cc66, emissiveIntensity: 0.4,
      roughness: 0.3, metalness: 0.4,
    });
    simFwdSphere = new THREE.Mesh(simGeo, simMat);
    scene.add(simFwdSphere);
    const wireMat = new THREE.LineBasicMaterial({ color: 0x00ff88, transparent: true, opacity: 0.4 });
    const wireGeo = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]);
    simFwdWireLine = new THREE.Line(wireGeo, wireMat);
    scene.add(simFwdWireLine);
  }

  // Sim hammer overlay — backward (blue)
  if (simHammerBwdData) {
    const simGeo = new THREE.SphereGeometry(0.07, 16, 16);
    const simMat = new THREE.MeshStandardMaterial({
      color: 0x4488ff, emissive: 0x2266cc, emissiveIntensity: 0.4,
      roughness: 0.3, metalness: 0.4,
    });
    simBwdSphere = new THREE.Mesh(simGeo, simMat);
    scene.add(simBwdSphere);
    const wireMat = new THREE.LineBasicMaterial({ color: 0x4488ff, transparent: true, opacity: 0.4 });
    const wireGeo = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]);
    simBwdWireLine = new THREE.Line(wireGeo, wireMat);
    scene.add(simBwdWireLine);
  }
}

function updateHammerFrame(frame) {
  const off = frame * 3;
  const [x, y, z] = camToThree(hammerData[off], hammerData[off + 1], hammerData[off + 2]);
  hammerSphere.position.set(x, y, z);

  // Hide hammer at invalid frames (NaN = post max-height, zero = legacy)
  const isInvalid = isNaN(hammerData[off]) || (hammerData[off] === 0 && hammerData[off+1] === 0 && hammerData[off+2] === 0);
  hammerSphere.visible = !isInvalid;

  // Wire from wrist midpoint to tracked hammer ball (hidden after release)
  // MHR70 skeleton: left_wrist=62, right_wrist=41
  const releaseFrame = metadata && metadata.throw_window && metadata.throw_window.release;
  const wireVisible = !isInvalid && (releaseFrame == null || frame <= releaseFrame);
  if (hammerWireLine && keypointsData) {
    if (wireVisible) {
      const KP_COUNT = 70;
      const L_WRIST = 62, R_WRIST = 41;
      const kpBase = frame * KP_COUNT * 3;
      const lx = keypointsData[kpBase + L_WRIST * 3];
      const ly = keypointsData[kpBase + L_WRIST * 3 + 1];
      const lz = keypointsData[kpBase + L_WRIST * 3 + 2];
      const rx = keypointsData[kpBase + R_WRIST * 3];
      const ry = keypointsData[kpBase + R_WRIST * 3 + 1];
      const rz = keypointsData[kpBase + R_WRIST * 3 + 2];
      const [wx, wy, wz] = camToThree((lx+rx)/2, (ly+ry)/2, (lz+rz)/2);
      const hPos = hammerWireLine.geometry.getAttribute('position');
      hPos.array[0] = wx; hPos.array[1] = wy; hPos.array[2] = wz;
      hPos.array[3] = x; hPos.array[4] = y; hPos.array[5] = z;
      hPos.needsUpdate = true;
      hammerWireLine.visible = true;
    } else {
      hammerWireLine.visible = false;
    }
  }

  // Color by fill_type provenance
  if (fillTypeData && !isInvalid) {
    const ft = fillTypeData[frame];
    const scheme = FILL_TYPE_COLORS[ft] || FILL_TYPE_COLORS[0];
    hammerSphere.material.color.setHex(scheme.color);
    hammerSphere.material.emissive.setHex(scheme.emissive);
  }

  // Sim hammer overlays (fwd=green, bwd=blue)
  // All wires connect to the same wrist midpoint (MHR70: L=62, R=41)
  // Wire hidden after release (hammer is no longer attached to hands)
  function updateSimBall(sphere, wireLine, simData) {
    if (!sphere || !simData) return;
    const sx = simData[off], sy = simData[off + 1], sz = simData[off + 2];
    const simInvalid = isNaN(sx) || (sx === 0 && sy === 0 && sz === 0);
    sphere.visible = !simInvalid;
    const simWireVisible = !simInvalid && (releaseFrame == null || frame <= releaseFrame);
    if (!simInvalid) {
      const [sx3, sy3, sz3] = camToThree(sx, sy, sz);
      sphere.position.set(sx3, sy3, sz3);
      if (wireLine && keypointsData && simWireVisible) {
        const KP_COUNT = 70;
        const L_WRIST = 62, R_WRIST = 41;
        const kpBase = frame * KP_COUNT * 3;
        const lx = keypointsData[kpBase + L_WRIST * 3];
        const ly = keypointsData[kpBase + L_WRIST * 3 + 1];
        const lz = keypointsData[kpBase + L_WRIST * 3 + 2];
        const rx = keypointsData[kpBase + R_WRIST * 3];
        const ry = keypointsData[kpBase + R_WRIST * 3 + 1];
        const rz = keypointsData[kpBase + R_WRIST * 3 + 2];
        const [wx, wy, wz] = camToThree((lx+rx)/2, (ly+ry)/2, (lz+rz)/2);
        const wPos = wireLine.geometry.getAttribute('position');
        wPos.array[0] = wx; wPos.array[1] = wy; wPos.array[2] = wz;
        wPos.array[3] = sx3; wPos.array[4] = sy3; wPos.array[5] = sz3;
        wPos.needsUpdate = true;
        wireLine.visible = true;
      } else if (wireLine) {
        wireLine.visible = false;
      }
    } else {
      if (wireLine) wireLine.visible = false;
    }
  }
  updateSimBall(simFwdSphere, simFwdWireLine, simHammerFwdData);
  updateSimBall(simBwdSphere, simBwdWireLine, simHammerBwdData);
}

// ─── Orbit Extremes (High/Low Points) ────────────────────────────────────────

function precomputeOrbitExtremes() {
  orbitExtremes = [];
  if (!hammerData || !metadata.turn_boundaries || metadata.turn_boundaries.length < 2) return;

  const tw = metadata.throw_window || {};
  const throwStart = tw.start || 0;
  const throwRelease = tw.release || (metadata.frame_count - 1);
  const boundaries = metadata.turn_boundaries;

  // Helper: get hammer Y at frame (Three.js Y = up)
  function hammerY(f) {
    const off = f * 3;
    if (isNaN(hammerData[off])) return null;
    if (hammerData[off] === 0 && hammerData[off + 1] === 0 && hammerData[off + 2] === 0) return null;
    return camToThree(hammerData[off], hammerData[off + 1], hammerData[off + 2])[1];
  }

  function hammerPos(f) {
    const off = f * 3;
    return camToThree(hammerData[off], hammerData[off + 1], hammerData[off + 2]);
  }

  // N turns (between consecutive boundaries) → N high points, N+1 low points
  // Step 1: Find one HIGH per turn segment (max Y between boundary[i] and boundary[i+1])
  const highFrames = [];
  for (let i = 0; i < boundaries.length - 1; i++) {
    const segStart = boundaries[i];
    const segEnd = boundaries[i + 1];
    let maxY = -Infinity, maxFrame = segStart;
    for (let f = segStart; f <= segEnd; f++) {
      const y = hammerY(f);
      if (y !== null && y > maxY) { maxY = y; maxFrame = f; }
    }
    if (maxY > -Infinity) {
      highFrames.push(maxFrame);
      const [hx, hy, hz] = hammerPos(maxFrame);
      orbitExtremes.push({ frame: maxFrame, pos: [hx, hy, hz], type: 'high', turnFrameCount: segEnd - segStart });
    }
  }

  // Step 2: Find N+1 LOW points in the gaps between highs
  // Segments: [pre-T0 start, high[0]], [high[0], high[1]], ..., [high[N-1], release]
  const halfTurn = boundaries.length >= 2 ? Math.round((boundaries[1] - boundaries[0]) / 2) : 10;
  const lowSearchStart = Math.max(0, boundaries[0] - halfTurn);
  const lowSegments = [];
  if (highFrames.length > 0) {
    lowSegments.push([lowSearchStart, highFrames[0]]);
    for (let i = 0; i < highFrames.length - 1; i++) {
      lowSegments.push([highFrames[i], highFrames[i + 1]]);
    }
    lowSegments.push([highFrames[highFrames.length - 1], throwRelease]);
  }

  for (const [segStart, segEnd] of lowSegments) {
    let minY = Infinity, minFrame = segStart;
    const turnFrameCount = segEnd - segStart;
    for (let f = segStart; f <= segEnd; f++) {
      const y = hammerY(f);
      if (y !== null && y < minY) { minY = y; minFrame = f; }
    }
    if (minY < Infinity) {
      const [lx, ly, lz] = hammerPos(minFrame);
      orbitExtremes.push({ frame: minFrame, pos: [lx, ly, lz], type: 'low', turnFrameCount });
    }
  }
}

function createOrbitExtremesSpheres() {
  orbitExtremesGroup = new THREE.Group();
  orbitExtremesGroup.visible = false;  // off until toggled

  for (const ext of orbitExtremes) {
    const geo = new THREE.SphereGeometry(0.08, 16, 16);
    const isHigh = ext.type === 'high';
    const mat = new THREE.MeshStandardMaterial({
      color: isHigh ? 0xff4444 : 0x44aaff,
      emissive: isHigh ? 0xcc2222 : 0x2288cc,
      emissiveIntensity: 0.3,
      roughness: 0.3,
      metalness: 0.4,
      transparent: true,
      opacity: 0,
    });
    const sphere = new THREE.Mesh(geo, mat);
    sphere.position.set(ext.pos[0], ext.pos[1], ext.pos[2]);
    sphere.visible = false;
    ext.sphere = sphere;
    orbitExtremesGroup.add(sphere);
  }

  scene.add(orbitExtremesGroup);
}

function updateOrbitExtremesFrame(frame) {
  if (!orbitExtremesGroup || !orbitExtremesGroup.visible) return;

  for (const ext of orbitExtremes) {
    if (!ext.sphere) continue;
    if (frame < ext.frame) {
      ext.sphere.visible = false;
    } else {
      const elapsed = frame - ext.frame;
      const fadeDuration = Math.round(0.75 * ext.turnFrameCount);
      ext.sphere.visible = true;
      if (fadeDuration <= 0 || elapsed > fadeDuration) {
        ext.sphere.material.opacity = 0.5;
      } else {
        ext.sphere.material.opacity = 1.0 - 0.5 * (elapsed / fadeDuration);
      }
    }
  }
}

// ─── Ground Reference ────────────────────────────────────────────────────────

function createGround() {
  // Grid at the feet level for spatial reference
  // We'll position it after loading the first frame
  groundGroup = new THREE.Group();
  scene.add(groundGroup);
}

function positionGround(frame) {
  // Clear old ground
  while (groundGroup.children.length) groundGroup.remove(groundGroup.children[0]);

  if (isWorldSpace) {
    // World space: ground_y from metadata, center grid on body using keypoints
    groundY = metadata.ground_y;
    const grid = new THREE.GridHelper(8, 16, 0x444444, 0x333333);
    grid.position.y = groundY;

    // Center grid on body centroid from keypoints (rigged viewer has no vertex data)
    if (keypointsData) {
      const nKp = 70;
      const kpOff = frame * nKp * 3;
      let cx = 0, cz = 0, count = 0;
      for (let i = 0; i < nKp; i++) {
        const x = keypointsData[kpOff + i * 3];
        const z = keypointsData[kpOff + i * 3 + 2];
        if (!isNaN(x) && !isNaN(z)) { cx += x; cz += z; count++; }
      }
      if (count > 0) { cx /= count; cz /= count; }
      grid.position.x = cx;
      grid.position.z = cz;
    }
    groundGroup.add(grid);
  } else {
    // Camera space fallback using keypoints
    groundY = metadata.ground_y || 0;
    const grid = new THREE.GridHelper(6, 12, 0x444444, 0x333333);
    grid.position.y = groundY;
    groundGroup.add(grid);
  }
}

// ─── Leg Planes ─────────────────────────────────────────────────────────────

function makeLegPlaneMesh(r, g, b) {
  const geo = new THREE.BufferGeometry();
  const pos = new Float32Array(4 * 3);
  const pa = new THREE.BufferAttribute(pos, 3);
  pa.setUsage(THREE.StreamDrawUsage);
  geo.setAttribute('position', pa);

  const uv = new Float32Array([0,1, 1,1, 0,0, 1,0]);
  geo.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
  geo.setIndex([0,2,1, 1,2,3]);

  const tex = createPlaneTexture(r, g, b);
  const mat = new THREE.MeshBasicMaterial({
    map: tex,
    transparent: true,
    opacity: 0.30,
    side: THREE.DoubleSide,
    depthWrite: false,
  });

  return { mesh: new THREE.Mesh(geo, mat), posAttr: pa, texture: tex };
}

function createLegPlanes() {
  legPlanesGroup = new THREE.Group();
  leftLegPlane  = makeLegPlaneMesh(70, 140, 220);   // blue
  rightLegPlane = makeLegPlaneMesh(220, 90, 70);     // coral
  legPlanesGroup.add(leftLegPlane.mesh);
  legPlanesGroup.add(rightLegPlane.mesh);
  legPlanesGroup.visible = false;
  scene.add(legPlanesGroup);
}

function updateOneLegPlane(frame, pl, hipIdx, kneeIdx, ankleIdx) {
  const hip   = getKp(frame, hipIdx);
  const knee  = getKp(frame, kneeIdx);
  const ankle = getKp(frame, ankleIdx);

  // Direction along the leg
  const mainDir = new THREE.Vector3().subVectors(ankle, hip).normalize();

  // Plane normal from hip-knee-ankle
  const v1 = new THREE.Vector3().subVectors(knee, hip);
  const v2 = new THREE.Vector3().subVectors(ankle, hip);
  const normal = new THREE.Vector3().crossVectors(v1, v2).normalize();

  // Width direction: in-plane, perpendicular to leg axis
  const widthDir = new THREE.Vector3().crossVectors(normal, mainDir).normalize();

  // Ensure width extends outward (away from body midline)
  const hipMid = getKp(frame, KP_LEFT_HIP).add(getKp(frame, KP_RIGHT_HIP)).multiplyScalar(0.5);
  const outward = new THREE.Vector3().subVectors(hip, hipMid);
  if (outward.dot(widthDir) < 0) widthDir.negate();

  // Extended ankle point
  const ankleExt = ankle.clone().addScaledVector(mainDir, PLANE_EXT);

  const wOff = widthDir.clone().multiplyScalar(PLANE_WIDTH);

  // Quad: A=hip, B=hip+w, D=ankleExt, C=ankleExt+w
  const verts = [
    [hip.x,           hip.y,           hip.z],
    [hip.x + wOff.x,  hip.y + wOff.y,  hip.z + wOff.z],
    [ankleExt.x,      ankleExt.y,      ankleExt.z],
    [ankleExt.x+wOff.x, ankleExt.y+wOff.y, ankleExt.z+wOff.z],
  ];

  // Clamp all vertices to ground level
  if (groundY > -Infinity) {
    for (const v of verts) {
      if (v[1] < groundY) v[1] = groundY;
    }
  }

  const a = pl.posAttr.array;
  for (let i = 0; i < 4; i++) {
    a[i*3] = verts[i][0]; a[i*3+1] = verts[i][1]; a[i*3+2] = verts[i][2];
  }
  pl.posAttr.needsUpdate = true;
  pl.mesh.geometry.computeVertexNormals();

  // Set grid density (~8 cm cells)
  const height = hip.distanceTo(ankleExt);
  const cellSize = 0.08;
  pl.texture.repeat.set(Math.max(1, PLANE_WIDTH / cellSize),
                        Math.max(1, height / cellSize));
}

function updateLegPlanes(frame) {
  // Hide leg planes once we reach T0 (first turn boundary)
  if (legPlanesGroup && legPlanesGroup.visible && metadata.turn_boundaries && metadata.turn_boundaries.length > 0) {
    if (frame >= metadata.turn_boundaries[0]) {
      legPlanesGroup.visible = false;
      const planesBtn = document.querySelector('.toggle-btn[data-target="planes"]');
      if (planesBtn) planesBtn.classList.remove('active');
    }
  }
  // Update graph cursor if graph is visible (timeline scrubbing updates the graph)
  const kaContainer = document.getElementById('kneeangle-container');
  if (kaContainer && kaContainer.style.display !== 'none' && legAlignmentData) {
    drawLegCorotationGraph(frame);
  }
  // Plane geometry is NOT updated per frame — planes are static at activation frame
}

// ─── Leg Co-rotation Graph ──────────────────────────────────────────────────

let legAlignmentData = null;  // Float32Array(T) — from analytics leg_alignment_deg

function drawLegCorotationGraph(frame) {
  const canvas = document.getElementById('kneeangle-graph');
  if (!canvas || !legAlignmentData) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width, h = canvas.height;

  // White background
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, w, h);

  const fStart = timelineMin;
  const fEnd = timelineMax;

  const pad = { left: 40, right: 10, top: 22, bottom: 22 };
  const plotW = w - pad.left - pad.right;
  const plotH = h - pad.top - pad.bottom;

  let dataMin = Infinity, dataMax = -Infinity;
  for (let i = fStart; i <= fEnd; i++) {
    const v = legAlignmentData[i];
    if (v < dataMin) dataMin = v;
    if (v > dataMax) dataMax = v;
  }
  const legBuf = Math.max((dataMax - dataMin) * 0.05, 1);
  const yMinVal = Math.floor((dataMin - legBuf) / 5) * 5;
  const yMaxVal = Math.ceil((dataMax + legBuf) / 5) * 5;
  const fRange = Math.max(1, fEnd - fStart);

  function xPx(f) { return pad.left + ((f - fStart) / fRange) * plotW; }
  function yPx(v) { return pad.top + (1 - (v - yMinVal) / (yMaxVal - yMinVal)) * plotH; }

  const inWindow = frame >= fStart && frame <= fEnd;

  // SS/DS shading (gray for SS, white for DS) + labels
  drawSSDSShading(ctx, xPx, pad, plotH, fStart, fEnd);

  // Reference thresholds
  ctx.setLineDash([6, 4]);
  ctx.lineWidth = 1;
  if (yMinVal <= 15 && yMaxVal >= 15) {
    ctx.strokeStyle = 'rgba(255, 165, 0, 0.7)';
    ctx.beginPath(); ctx.moveTo(pad.left, yPx(15)); ctx.lineTo(w - pad.right, yPx(15)); ctx.stroke();
  }
  if (yMinVal <= 30 && yMaxVal >= 30) {
    ctx.strokeStyle = 'rgba(220, 50, 50, 0.7)';
    ctx.beginPath(); ctx.moveTo(pad.left, yPx(30)); ctx.lineTo(w - pad.right, yPx(30)); ctx.stroke();
  }
  ctx.setLineDash([]);

  // Y-axis labels + light grid
  ctx.fillStyle = '#444';
  ctx.font = '10px sans-serif';
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  const legStep = (yMaxVal - yMinVal) <= 30 ? 5 : 10;
  for (let v = yMinVal; v <= yMaxVal; v += legStep) {
    ctx.fillText(v + '\u00B0', pad.left - 4, yPx(v));
    if (v > yMinVal && v < yMaxVal) {
      ctx.strokeStyle = 'rgba(0,0,0,0.08)';
      ctx.lineWidth = 0.5;
      ctx.beginPath(); ctx.moveTo(pad.left, yPx(v)); ctx.lineTo(w - pad.right, yPx(v)); ctx.stroke();
    }
  }

  // Turn boundaries + release
  drawTurnMarkers(ctx, xPx, pad, plotH, fStart, fEnd);

  // Title
  ctx.fillStyle = '#222';
  ctx.font = 'bold 11px sans-serif';
  ctx.textAlign = 'left';
  ctx.textBaseline = 'top';
  ctx.fillText('Leg Co-rotation', pad.left, 3);

  // Current value
  if (inWindow) {
    ctx.fillStyle = '#228B22';
    ctx.textAlign = 'right';
    ctx.fillText(legAlignmentData[frame].toFixed(1) + '\u00B0', w - pad.right, 3);
  }

  // Plot line — DS bold green, SS faded gray
  for (let i = fStart + 1; i <= fEnd; i++) {
    const isDS = supportStateData ? (supportStateData[i] === 2) : true;
    ctx.strokeStyle = isDS ? '#228B22' : 'rgba(180, 180, 180, 0.7)';
    ctx.lineWidth = isDS ? 2.5 : 1.0;
    ctx.beginPath();
    ctx.moveTo(xPx(i - 1), yPx(legAlignmentData[i - 1]));
    ctx.lineTo(xPx(i), yPx(legAlignmentData[i]));
    ctx.stroke();
  }

  // Cursor line — dark for visibility on white
  if (inWindow) {
    ctx.strokeStyle = 'rgba(0, 0, 0, 0.7)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(xPx(frame), pad.top);
    ctx.lineTo(xPx(frame), pad.top + plotH);
    ctx.stroke();
  }
}

// ─── Back Tilt Plane + Graph ─────────────────────────────────────────────────

let backTiltAngles = null;  // Float32Array(T) — pre-computed signed tilt per frame

function computeBackTiltAngle(frame) {
  const lShoulder = getKp(frame, KP_LEFT_SHOULDER);
  const rShoulder = getKp(frame, KP_RIGHT_SHOULDER);
  const lHip = getKp(frame, KP_LEFT_HIP);
  const rHip = getKp(frame, KP_RIGHT_HIP);

  const hipMid = lHip.clone().add(rHip).multiplyScalar(0.5);
  const shoulderMid = lShoulder.clone().add(rShoulder).multiplyScalar(0.5);

  // Tilt magnitude: angle between torso vector and vertical
  const torsoDir = new THREE.Vector3().subVectors(shoulderMid, hipMid).normalize();
  const up = new THREE.Vector3(0, 1, 0);
  const dot = Math.min(1, Math.max(-1, torsoDir.dot(up)));
  const tiltRad = Math.acos(dot);  // 0 = vertical, π/2 = horizontal

  // Sign: positive = leaning AWAY from hammer (back lean)
  const shoulderDisp = new THREE.Vector3().subVectors(shoulderMid, hipMid);
  shoulderDisp.y = 0;

  const hOff = frame * 3;
  const hammerValid = hammerData && !isNaN(hammerData[hOff]) &&
    !(hammerData[hOff] === 0 && hammerData[hOff + 1] === 0 && hammerData[hOff + 2] === 0);

  let sign = 1;
  if (hammerValid) {
    const [hx, hy, hz] = camToThree(hammerData[hOff], hammerData[hOff + 1], hammerData[hOff + 2]);
    const hammerDir = new THREE.Vector3(hx - hipMid.x, 0, hz - hipMid.z);
    if (hammerDir.lengthSq() > 0.001) {
      hammerDir.normalize();
      sign = shoulderDisp.dot(hammerDir) < 0 ? 1 : -1;
    }
  }

  return sign * tiltRad * (180 / Math.PI);
}

function precomputeBackTilt() {
  const T = metadata.frame_count;
  if (pipelineBackLean && pipelineBackLean.length === T) {
    backTiltAngles = pipelineBackLean;
    console.log('Using pipeline-precomputed back lean angles');
    return;
  }
  // Fallback: compute live from keypoints (less accurate)
  backTiltAngles = new Float32Array(T);
  for (let i = 0; i < T; i++) {
    backTiltAngles[i] = computeBackTiltAngle(i);
  }
  console.log('Using live-computed back tilt angles (no pipeline data)');
}

function drawSSDSShading(ctx, xPx, pad, plotH, fStart, fEnd) {
  if (!supportStateData) return;
  // Draw contiguous SS regions as gray shading (DS = white, SS = gray)
  let regionStart = fStart;
  let regionState = supportStateData[fStart];
  for (let i = fStart + 1; i <= fEnd + 1; i++) {
    const s = i <= fEnd ? supportStateData[i] : -1;
    if (s !== regionState || i > fEnd) {
      const x0 = xPx(regionStart - 0.5);
      const x1 = xPx(i - 0.5);
      if (regionState === 1) {
        // SS: light gray shading
        ctx.fillStyle = 'rgba(0, 0, 0, 0.07)';
        ctx.fillRect(x0, pad.top, x1 - x0, plotH);
      }
      // Label at bottom of region (skip value 0 = outside throw window)
      if (regionState === 1 || regionState === 2) {
        const midX = (x0 + x1) / 2;
        const label = regionState === 1 ? 'SS' : 'DS';
        ctx.fillStyle = '#999';
        ctx.font = 'bold 8px sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'bottom';
        ctx.fillText(label, midX, pad.top + plotH - 1);
      }
      regionStart = i;
      regionState = s;
    }
  }
}

function drawTurnMarkers(ctx, xPx, pad, plotH, fStart, fEnd) {
  if (!metadata.turn_boundaries) return;
  ctx.save();
  ctx.setLineDash([4, 4]);
  ctx.lineWidth = 1;
  ctx.font = '9px sans-serif';
  ctx.textAlign = 'center';

  metadata.turn_boundaries.forEach((f, i) => {
    if (f < fStart || f > fEnd) return;
    const x = xPx(f);
    ctx.strokeStyle = 'rgba(100, 150, 200, 0.5)';
    ctx.beginPath();
    ctx.moveTo(x, pad.top);
    ctx.lineTo(x, pad.top + plotH);
    ctx.stroke();
    const label = metadata.turn_labels ? metadata.turn_labels[i] : `T${i}`;
    ctx.fillStyle = '#666';
    ctx.textBaseline = 'top';
    ctx.fillText(label, x, pad.top + 2);
  });

  // Release marker — blue dashed
  const relFrame = metadata.throw_window && metadata.throw_window.release;
  if (relFrame && relFrame >= fStart && relFrame <= fEnd) {
    const x = xPx(relFrame);
    ctx.setLineDash([6, 3]);
    ctx.strokeStyle = 'rgba(50, 100, 200, 0.8)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(x, pad.top);
    ctx.lineTo(x, pad.top + plotH);
    ctx.stroke();
    ctx.fillStyle = 'rgba(50, 100, 200, 0.9)';
    ctx.font = 'bold 9px sans-serif';
    ctx.textBaseline = 'top';
    ctx.fillText('REL', x, pad.top + 2);
  }

  ctx.restore();
}

function drawBackTiltGraph(frame) {
  const canvas = document.getElementById('backtilt-graph');
  if (!canvas || !backTiltAngles) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width, h = canvas.height;

  // White background
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, w, h);

  const fStart = timelineMin;
  const fEnd = timelineMax;

  const pad = { left: 40, right: 10, top: 22, bottom: 22 };
  const plotW = w - pad.left - pad.right;
  const plotH = h - pad.top - pad.bottom;

  let dataMin = Infinity, dataMax = -Infinity;
  for (let i = fStart; i <= fEnd; i++) {
    const v = backTiltAngles[i];
    if (v < dataMin) dataMin = v;
    if (v > dataMax) dataMax = v;
  }
  const btBuf = Math.max((dataMax - dataMin) * 0.05, 1);
  const yMin = Math.floor((dataMin - btBuf) / 5) * 5;
  const yMax = Math.ceil((dataMax + btBuf) / 5) * 5;
  const fRange = Math.max(1, fEnd - fStart);

  function xPx(f) { return pad.left + ((f - fStart) / fRange) * plotW; }
  function yPx(v) { return pad.top + (1 - (v - yMin) / (yMax - yMin)) * plotH; }

  const curTilt = backTiltAngles[frame];
  const inWindow = frame >= fStart && frame <= fEnd;

  // SS/DS shading
  drawSSDSShading(ctx, xPx, pad, plotH, fStart, fEnd);

  // Zero line
  if (yMin <= 0 && yMax >= 0) {
    ctx.strokeStyle = 'rgba(0, 0, 0, 0.3)';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(pad.left, yPx(0));
    ctx.lineTo(w - pad.right, yPx(0));
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // Y-axis labels + light grid
  ctx.fillStyle = '#444';
  ctx.font = '10px sans-serif';
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  const btStep = (yMax - yMin) <= 30 ? 5 : 10;
  for (let v = yMin; v <= yMax; v += btStep) {
    ctx.fillText(v + '\u00B0', pad.left - 4, yPx(v));
    if (v !== 0 && v > yMin && v < yMax) {
      ctx.strokeStyle = 'rgba(0,0,0,0.08)';
      ctx.lineWidth = 0.5;
      ctx.beginPath(); ctx.moveTo(pad.left, yPx(v)); ctx.lineTo(w - pad.right, yPx(v)); ctx.stroke();
    }
  }

  // Turn boundaries + release
  drawTurnMarkers(ctx, xPx, pad, plotH, fStart, fEnd);

  // Title
  ctx.fillStyle = '#222';
  ctx.font = 'bold 11px sans-serif';
  ctx.textAlign = 'left';
  ctx.textBaseline = 'top';
  ctx.fillText('Back Tilt', pad.left, 3);

  // Current value
  if (inWindow) {
    const valColor = curTilt >= 0 ? '#228B22' : '#cc3333';
    ctx.fillStyle = valColor;
    ctx.textAlign = 'right';
    ctx.fillText(curTilt.toFixed(1) + '\u00B0', w - pad.right, 3);
  }

  // Plot line — green for positive (back), red for negative (forward)
  ctx.lineWidth = 2.0;
  for (let i = fStart + 1; i <= fEnd; i++) {
    const v0 = backTiltAngles[i - 1], v1 = backTiltAngles[i];
    ctx.strokeStyle = ((v0 + v1) / 2) >= 0 ? '#228B22' : '#cc3333';
    ctx.beginPath();
    ctx.moveTo(xPx(i - 1), yPx(v0));
    ctx.lineTo(xPx(i), yPx(v1));
    ctx.stroke();
  }

  // Cursor line
  if (inWindow) {
    ctx.strokeStyle = 'rgba(0, 0, 0, 0.7)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(xPx(frame), pad.top);
    ctx.lineTo(xPx(frame), pad.top + plotH);
    ctx.stroke();
  }
}

function createBackPlane() {
  backPlanesGroup = new THREE.Group();

  // Quad for torso (green tint)
  backPlane = makeLegPlaneMesh(70, 180, 120);
  backPlanesGroup.add(backPlane.mesh);

  backPlanesGroup.visible = false;
  scene.add(backPlanesGroup);

  // Pre-compute tilt angles for graph
  precomputeBackTilt();
}

function updateBackPlane(frame) {
  // Always update graph if angles exist and container is visible
  const container = document.getElementById('backtilt-container');
  if (container && container.style.display !== 'none' && backTiltAngles) {
    drawBackTiltGraph(frame);
  }

  if (!backPlanesGroup || !backPlanesGroup.visible) return;

  const lShoulder = getKp(frame, KP_LEFT_SHOULDER);
  const rShoulder = getKp(frame, KP_RIGHT_SHOULDER);
  const lHip = getKp(frame, KP_LEFT_HIP);
  const rHip = getKp(frame, KP_RIGHT_HIP);

  const hipMid = lHip.clone().add(rHip).multiplyScalar(0.5);
  const shoulderMid = lShoulder.clone().add(rShoulder).multiplyScalar(0.5);

  // Compute plane normal pointing backward (away from chest)
  const shoulderVec = new THREE.Vector3().subVectors(rShoulder, lShoulder);
  const torsoVec = new THREE.Vector3().subVectors(hipMid, shoulderMid);
  const planeNormal = new THREE.Vector3().crossVectors(shoulderVec, torsoVec).normalize();

  // Ensure normal points backward: compare with direction toward nose (KP 0 = nose)
  const nose = getKp(frame, 0);
  const torsoCenter = shoulderMid.clone().add(hipMid).multiplyScalar(0.5);
  const toNose = new THREE.Vector3().subVectors(nose, torsoCenter);
  if (planeNormal.dot(toNose) > 0) planeNormal.negate();

  // Offset vertices behind the mesh along backward normal
  const backOff = 0.15;  // 15cm behind mesh surface
  const lsBack = lShoulder.clone().addScaledVector(planeNormal, backOff);
  const rsBack = rShoulder.clone().addScaledVector(planeNormal, backOff);
  const lhBack = lHip.clone().addScaledVector(planeNormal, backOff);
  const rhBack = rHip.clone().addScaledVector(planeNormal, backOff);

  // Set quad corners: 0=lShoulder, 1=rShoulder, 2=lHip, 3=rHip
  const a = backPlane.posAttr.array;
  a[0] = lsBack.x; a[1] = lsBack.y; a[2] = lsBack.z;
  a[3] = rsBack.x; a[4] = rsBack.y; a[5] = rsBack.z;
  a[6] = lhBack.x; a[7] = lhBack.y; a[8] = lhBack.z;
  a[9] = rhBack.x; a[10] = rhBack.y; a[11] = rhBack.z;
  backPlane.posAttr.needsUpdate = true;
  backPlane.mesh.geometry.computeVertexNormals();
  backPlane.mesh.geometry.computeBoundingSphere();

  // Grid density
  const height = shoulderMid.distanceTo(hipMid);
  const width = lShoulder.distanceTo(rShoulder);
  const cellSize = 0.08;
  backPlane.texture.repeat.set(Math.max(1, width / cellSize), Math.max(1, height / cellSize));
}

// ─── Hip-Shoulder Separation ─────────────────────────────────────────────────

function computeSeparationAngle(frame) {
  const lHip = getKp(frame, KP_LEFT_HIP);
  const rHip = getKp(frame, KP_RIGHT_HIP);
  const lShoulder = getKp(frame, KP_LEFT_SHOULDER);
  const rShoulder = getKp(frame, KP_RIGHT_SHOULDER);

  // Project hip and shoulder vectors onto XZ plane
  const hipVecX = rHip.x - lHip.x;
  const hipVecZ = rHip.z - lHip.z;
  const shoulderVecX = rShoulder.x - lShoulder.x;
  const shoulderVecZ = rShoulder.z - lShoulder.z;

  // Signed angle via atan2(cross, dot)
  const dot = hipVecX * shoulderVecX + hipVecZ * shoulderVecZ;
  const cross = hipVecX * shoulderVecZ - hipVecZ * shoulderVecX;
  return Math.atan2(cross, dot) * (180 / Math.PI);
}

function precomputeSeparation() {
  const T = metadata.frame_count;
  if (pipelineSeparation && pipelineSeparation.length === T) {
    separationAngles = pipelineSeparation;
    console.log('Using pipeline-precomputed separation angles');
    return;
  }
  // Fallback: compute live from keypoints (less accurate)
  separationAngles = new Float32Array(T);
  for (let i = 0; i < T; i++) {
    separationAngles[i] = computeSeparationAngle(i);
  }
  console.log('Using live-computed separation angles (no pipeline data)');
}

const SKIN_R = 0.2, SKIN_G = 0.4, SKIN_B = 0.8;
const GREEN_R = 0.2, GREEN_G = 1.0, GREEN_B = 0.2;
const RED_R = 1.0, RED_G = 0.2, RED_B = 0.2;

function updateTorsoColors(frame) {
  const mesh = bodyMesh || riggedModel;
  if (!mesh) return;
  const colorAttr = mesh.geometry.getAttribute('color');
  if (!colorAttr) return;
  const arr = colorAttr.array;
  const V = colorAttr.count;

  // Base color depends on mode
  const baseR = bodyMesh ? SKIN_R : BASE_R;
  const baseG = bodyMesh ? SKIN_G : BASE_G;
  const baseB = bodyMesh ? SKIN_B : BASE_B;

  if (!separationEnabled || !separationAngles) {
    // Reset to base colors
    if (bodyMesh && vertexColorsData && vertexColorsData.length === V * 3) {
      arr.set(vertexColorsData);
    } else {
      for (let i = 0; i < V; i++) {
        arr[i * 3]     = baseR;
        arr[i * 3 + 1] = baseG;
        arr[i * 3 + 2] = baseB;
      }
    }
    colorAttr.needsUpdate = true;
    return;
  }

  // Compute separation color
  const angle = separationAngles[frame];
  let fraction, tR, tG, tB;
  if (angle >= 0) {
    fraction = Math.min(angle / 45, 1);
    tR = baseR + (GREEN_R - baseR) * fraction;
    tG = baseG + (GREEN_G - baseG) * fraction;
    tB = baseB + (GREEN_B - baseB) * fraction;
  } else {
    fraction = Math.min(-angle / 15, 1);
    tR = baseR + (RED_R - baseR) * fraction;
    tG = baseG + (RED_G - baseG) * fraction;
    tB = baseB + (RED_B - baseB) * fraction;
  }

  if (bodyMesh) {
    // Original vertex mesh: Y-range coloring
    const lHip = getKp(frame, KP_LEFT_HIP);
    const rHip = getKp(frame, KP_RIGHT_HIP);
    const lShoulder = getKp(frame, KP_LEFT_SHOULDER);
    const rShoulder = getKp(frame, KP_RIGHT_SHOULDER);
    const hipMidY = (lHip.y + rHip.y) * 0.5;
    const shoulderMidY = (lShoulder.y + rShoulder.y) * 0.5;
    const boundary = hipMidY + fraction * (shoulderMidY - hipMidY);
    const yLow = Math.min(hipMidY, boundary);
    const yHigh = Math.max(hipMidY, boundary);
    const posAttr = bodyMesh.geometry.getAttribute('position');
    const posArr = posAttr.array;

    for (let i = 0; i < V; i++) {
      const vy = posArr[i * 3 + 1];
      if (vy >= yLow && vy <= yHigh) {
        arr[i * 3]     = tR;
        arr[i * 3 + 1] = tG;
        arr[i * 3 + 2] = tB;
      } else if (vertexColorsData && vertexColorsData.length === V * 3) {
        arr[i * 3]     = vertexColorsData[i * 3];
        arr[i * 3 + 1] = vertexColorsData[i * 3 + 1];
        arr[i * 3 + 2] = vertexColorsData[i * 3 + 2];
      } else {
        arr[i * 3]     = SKIN_R;
        arr[i * 3 + 1] = SKIN_G;
        arr[i * 3 + 2] = SKIN_B;
      }
    }
  } else if (riggedModel && torsoMask) {
    // Rigged: Y-range growing (same logic as regular viewer)
    // Approximate skinned vertex Y from weighted bone world positions
    const lHip = getKp(frame, KP_LEFT_HIP);
    const rHip = getKp(frame, KP_RIGHT_HIP);
    const lShoulder = getKp(frame, KP_LEFT_SHOULDER);
    const rShoulder = getKp(frame, KP_RIGHT_SHOULDER);
    const hipMidY = (lHip.y + rHip.y) * 0.5;
    const shoulderMidY = (lShoulder.y + rShoulder.y) * 0.5;
    const boundary = hipMidY + fraction * (shoulderMidY - hipMidY);
    const yLow = Math.min(hipMidY, boundary);
    const yHigh = Math.max(hipMidY, boundary);

    // Precompute bone world Y (once per frame, ~52 bones)
    const bones = riggedSkeleton.bones;
    const boneWorldY = new Float32Array(bones.length);
    const _tmpV = new THREE.Vector3();
    for (let b = 0; b < bones.length; b++) {
      bones[b].getWorldPosition(_tmpV);
      boneWorldY[b] = _tmpV.y;
    }

    const skinIdx = riggedModel.geometry.getAttribute('skinIndex');
    const skinWt = riggedModel.geometry.getAttribute('skinWeight');
    for (let i = 0; i < V; i++) {
      if (torsoMask[i]) {
        // Approximate this vertex's world Y from bone weights
        let vy = 0;
        for (let j = 0; j < 4; j++) {
          vy += boneWorldY[skinIdx.getComponent(i, j)] * skinWt.getComponent(i, j);
        }
        if (vy >= yLow && vy <= yHigh) {
          arr[i * 3]     = tR;
          arr[i * 3 + 1] = tG;
          arr[i * 3 + 2] = tB;
        } else {
          arr[i * 3]     = baseR;
          arr[i * 3 + 1] = baseG;
          arr[i * 3 + 2] = baseB;
        }
      } else {
        arr[i * 3]     = baseR;
        arr[i * 3 + 1] = baseG;
        arr[i * 3 + 2] = baseB;
      }
    }
  }
  colorAttr.needsUpdate = true;
}

function drawSeparationGraph(frame) {
  const canvas = document.getElementById('separation-graph');
  if (!canvas || !separationAngles) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width, h = canvas.height;

  // White background
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, w, h);

  const fStart = timelineMin;
  const fEnd = timelineMax;

  const pad = { left: 40, right: 10, top: 22, bottom: 22 };
  const plotW = w - pad.left - pad.right;
  const plotH = h - pad.top - pad.bottom;

  let dataMin = Infinity, dataMax = -Infinity;
  for (let i = fStart; i <= fEnd; i++) {
    const v = separationAngles[i];
    if (v < dataMin) dataMin = v;
    if (v > dataMax) dataMax = v;
  }
  const sepBuf = Math.max((dataMax - dataMin) * 0.05, 1);
  const yMin = Math.floor((dataMin - sepBuf) / 5) * 5;
  const yMax = Math.ceil((dataMax + sepBuf) / 5) * 5;
  const fRange = Math.max(1, fEnd - fStart);

  function xPx(f) { return pad.left + ((f - fStart) / fRange) * plotW; }
  function yPx(v) { return pad.top + (1 - (v - yMin) / (yMax - yMin)) * plotH; }

  const curAngle = separationAngles[frame];
  const inWindow = frame >= fStart && frame <= fEnd;

  // SS/DS shading
  drawSSDSShading(ctx, xPx, pad, plotH, fStart, fEnd);

  // Zero line
  if (yMin <= 0 && yMax >= 0) {
    ctx.strokeStyle = 'rgba(0, 0, 0, 0.3)';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(pad.left, yPx(0));
    ctx.lineTo(w - pad.right, yPx(0));
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // Y-axis labels + light grid
  ctx.fillStyle = '#444';
  ctx.font = '10px sans-serif';
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  const sepStep = (yMax - yMin) <= 30 ? 5 : 10;
  for (let v = yMin; v <= yMax; v += sepStep) {
    ctx.fillText(v + '\u00B0', pad.left - 4, yPx(v));
    if (v !== 0 && v > yMin && v < yMax) {
      ctx.strokeStyle = 'rgba(0,0,0,0.08)';
      ctx.lineWidth = 0.5;
      ctx.beginPath(); ctx.moveTo(pad.left, yPx(v)); ctx.lineTo(w - pad.right, yPx(v)); ctx.stroke();
    }
  }

  // Turn boundaries + release
  drawTurnMarkers(ctx, xPx, pad, plotH, fStart, fEnd);

  // Title
  ctx.fillStyle = '#222';
  ctx.font = 'bold 11px sans-serif';
  ctx.textAlign = 'left';
  ctx.textBaseline = 'top';
  ctx.fillText('Separation', pad.left, 3);

  // Current value
  if (inWindow) {
    ctx.fillStyle = '#6A5ACD';
    ctx.textAlign = 'right';
    ctx.fillText(curAngle.toFixed(1) + '\u00B0', w - pad.right, 3);
  }

  // Plot line — purple (matching analytics style)
  ctx.strokeStyle = '#6A5ACD';
  ctx.lineWidth = 2.0;
  for (let i = fStart + 1; i <= fEnd; i++) {
    ctx.beginPath();
    ctx.moveTo(xPx(i - 1), yPx(separationAngles[i - 1]));
    ctx.lineTo(xPx(i), yPx(separationAngles[i]));
    ctx.stroke();
  }

  // Cursor line
  if (inWindow) {
    ctx.strokeStyle = 'rgba(0, 0, 0, 0.7)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(xPx(frame), pad.top);
    ctx.lineTo(xPx(frame), pad.top + plotH);
    ctx.stroke();
  }
}

// ─── Throwing Circle ──────────────────────────────────────────────────────────

function createCircle() {
  circleGroup = new THREE.Group();
  circleGroup.visible = false;  // hidden until data loaded

  if (!metadata.circle || !metadata.circle.detected) return;

  const c = metadata.circle;
  const radius = c.radius;
  const segments = 64;

  // Circle geometry in XZ plane (Y-up in Three.js = ground plane)
  const points = [];
  for (let i = 0; i <= segments; i++) {
    const theta = (i / segments) * Math.PI * 2;
    points.push(new THREE.Vector3(
      Math.cos(theta) * radius,
      0,
      Math.sin(theta) * radius,
    ));
  }
  const geometry = new THREE.BufferGeometry().setFromPoints(points);
  const material = new THREE.LineBasicMaterial({
    color: 0xffffff,
    linewidth: 2,
    transparent: true,
    opacity: 0.7,
  });
  circleLine = new THREE.Line(geometry, material);

  if (isWorldSpace && c.center) {
    // World space: fixed position from metadata
    circleLine.position.set(c.center[0], c.center[1], c.center[2]);
  } else if (c.center_cam) {
    // Camera space: flip (x, -y, -z)
    circleLine.position.set(c.center_cam[0], -c.center_cam[1], -c.center_cam[2]);
  }

  circleGroup.add(circleLine);
  circleGroup.visible = true;
  scene.add(circleGroup);
}

function updateCircleFrame(frame) {
  // World space: circle is fixed, no per-frame update needed
  if (isWorldSpace) return;

  if (!circleLine || !circlePositionsData) return;
  if (!metadata.circle || !metadata.circle.per_frame) return;

  const off = frame * 3;
  const cx = circlePositionsData[off];
  const cy = circlePositionsData[off + 1];
  const cz = circlePositionsData[off + 2];
  circleLine.position.set(cx, -cy, -cz);
  if (circleLabelsGroup) circleLabelsGroup.position.set(cx, -cy, -cz);
}

// ─── Circle Degree Labels ────────────────────────────────────────────────────

function createCircleLabels() {
  if (!circleGroup || !circleLine) return;
  if (!metadata.circle || !metadata.circle.detected) return;

  const radius = metadata.circle.radius;
  const center = circleLine.position;

  // "Toward camera" direction projected onto ground plane (XZ)
  let towardCam;
  if (isWorldSpace) {
    const cp = metadata.camera_position;
    const cc = metadata.circle.center || [0, 0, 0];
    towardCam = new THREE.Vector3(cp[0] - cc[0], 0, cp[2] - cc[2]).normalize();
  } else {
    towardCam = new THREE.Vector3(-center.x, 0, -center.z).normalize();
  }

  const up = new THREE.Vector3(0, 1, 0);
  const rightDir = new THREE.Vector3().crossVectors(up, towardCam).normalize();

  const labels = [
    { text: '0\u00B0',   dir: towardCam,                   isZero: true },
    { text: '90\u00B0',  dir: rightDir,                     isZero: false },
    { text: '180\u00B0', dir: towardCam.clone().negate(),    isZero: false },
    { text: '270\u00B0', dir: rightDir.clone().negate(),     isZero: false },
  ];

  const labelLen = 0.9;
  const gap = 0.05;
  const labelOffset = radius + gap + labelLen / 2;

  const camYAngle = Math.atan2(towardCam.x, towardCam.z);

  // Separate group for labels — rotatable independently of the ring
  circleLabelsGroup = new THREE.Group();
  circleLabelsGroup.position.copy(center);

  for (const { text, dir, isZero } of labels) {
    const canvas = document.createElement('canvas');
    canvas.width = 128;
    canvas.height = 48;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, 128, 48);
    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 40px sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, 64, 24);

    const texture = new THREE.CanvasTexture(canvas);
    const geo = new THREE.PlaneGeometry(labelLen, labelLen * 48 / 128);
    const mat = new THREE.MeshBasicMaterial({
      map: texture,
      transparent: true,
      side: THREE.DoubleSide,
      depthWrite: false,
    });
    const mesh = new THREE.Mesh(geo, mat);

    mesh.position.set(dir.x * labelOffset, 0.005, dir.z * labelOffset);
    mesh.rotation.x = -Math.PI / 2;
    mesh.rotation.z = camYAngle;

    circleLabelsGroup.add(mesh);

    if (isZero) labelZeroMesh = mesh;
  }

  circleGroup.add(circleLabelsGroup);
}

// ─── Label Drag Interaction ──────────────────────────────────────────────────

function setupLabelDrag() {
  if (!circleLabelsGroup || !labelZeroMesh) return;

  const domEl = renderer.domElement;
  const pointerNDC = new THREE.Vector2();
  const intersectPoint = new THREE.Vector3();

  function updateGroundPlane() {
    // Horizontal plane at the label group's world Y
    const worldPos = new THREE.Vector3();
    circleLabelsGroup.getWorldPosition(worldPos);
    labelGroundPlane.set(new THREE.Vector3(0, 1, 0), -worldPos.y);
  }

  function getNDC(event) {
    const rect = domEl.getBoundingClientRect();
    pointerNDC.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    pointerNDC.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
  }

  function getGroundPoint(event) {
    getNDC(event);
    labelRaycaster.setFromCamera(pointerNDC, camera);
    const hit = labelRaycaster.ray.intersectPlane(labelGroundPlane, intersectPoint);
    return hit;
  }

  domEl.addEventListener('pointerdown', (event) => {
    if (!circleLabelsGroup) return;

    getNDC(event);
    labelRaycaster.setFromCamera(pointerNDC, camera);
    const hits = labelRaycaster.intersectObjects(circleLabelsGroup.children);

    // Only the 0° label initiates drag
    if (hits.length > 0 && hits[0].object === labelZeroMesh) {
      isDraggingLabel = true;

      updateGroundPlane();
      const gp = getGroundPoint(event);
      if (!gp) { isDraggingLabel = false; return; }

      const center = new THREE.Vector3();
      circleLabelsGroup.getWorldPosition(center);
      labelDragStartAngle = Math.atan2(gp.x - center.x, gp.z - center.z);
      labelDragStartRotation = circleLabelsGroup.rotation.y;

      controls.enabled = false;
      domEl.style.cursor = 'grabbing';
    }
  });

  domEl.addEventListener('pointermove', (event) => {
    if (isDraggingLabel) {
      const gp = getGroundPoint(event);
      if (!gp) return;

      const center = new THREE.Vector3();
      circleLabelsGroup.getWorldPosition(center);
      const currentAngle = Math.atan2(gp.x - center.x, gp.z - center.z);
      circleLabelsGroup.rotation.y = labelDragStartRotation + (currentAngle - labelDragStartAngle);
      return;
    }

    // Hover feedback: show grab cursor over 0° label
    if (!circleLabelsGroup) return;
    getNDC(event);
    labelRaycaster.setFromCamera(pointerNDC, camera);
    const hits = labelRaycaster.intersectObjects(circleLabelsGroup.children);
    if (hits.length > 0 && hits[0].object === labelZeroMesh) {
      domEl.style.cursor = 'grab';
    } else if (domEl.style.cursor === 'grab') {
      domEl.style.cursor = '';
    }
  });

  const endDrag = () => {
    if (!isDraggingLabel) return;
    isDraggingLabel = false;
    controls.enabled = true;
    domEl.style.cursor = '';
  };

  domEl.addEventListener('pointerup', endDrag);
  domEl.addEventListener('pointercancel', endDrag);
}

// ─── Poles ───────────────────────────────────────────────────────────────────

function createPoles() {
  poleGroup = new THREE.Group();
  poleGroup.visible = false;  // hidden until toggled on

  if (!metadata.poles || metadata.poles.count === 0) return;

  const poleColor = 0x88ccff;
  const poleMaterial = new THREE.LineBasicMaterial({
    color: poleColor,
    linewidth: 2,
    transparent: true,
    opacity: 0.8,
  });

  if (isWorldSpace && metadata.poles.positions) {
    // World space: static vertical lines at known positions
    const height = metadata.poles.height || 3.0;
    const gy = metadata.ground_y || 0;

    for (const pos of metadata.poles.positions) {
      const points = [
        new THREE.Vector3(pos[0], gy, pos[2]),
        new THREE.Vector3(pos[0], gy + height, pos[2]),
      ];
      const geom = new THREE.BufferGeometry().setFromPoints(points);
      const line = new THREE.Line(geom, poleMaterial.clone());
      poleGroup.add(line);
    }
  } else if (polePositionsData && metadata.poles.per_frame) {
    // Camera space: create placeholder lines, updated per-frame
    const nPoles = metadata.poles.count;
    for (let i = 0; i < nPoles; i++) {
      const points = [
        new THREE.Vector3(0, 0, 0),
        new THREE.Vector3(0, 1, 0),
      ];
      const geom = new THREE.BufferGeometry().setFromPoints(points);
      geom.setAttribute('position', new THREE.Float32BufferAttribute(
        [0, 0, 0, 0, 1, 0], 3
      ));
      const line = new THREE.Line(geom, poleMaterial.clone());
      line.visible = false;
      poleGroup.add(line);
    }
  }

  scene.add(poleGroup);
}

function updatePolesFrame(frame) {
  if (!poleGroup || !poleGroup.visible) return;

  // World space: poles are static, no per-frame update
  if (isWorldSpace) return;

  // Camera space: update from per-frame 2D positions
  if (!polePositionsData || !metadata.poles || !metadata.poles.per_frame) return;

  const nPoles = metadata.poles.count;
  const T = metadata.frame_count;

  // Approximate body depth from first vertex Z
  const V = metadata.vertex_count;
  const vOff = frame * V * 3;
  let avgZ = 0;
  for (let i = 0; i < Math.min(V, 100); i++) {
    avgZ += verticesData[vOff + i * 3 + 2];
  }
  avgZ /= Math.min(V, 100);
  const depth = Math.abs(avgZ);

  const focalLength = metadata.focal_length || 2200;
  const poleHeightPx = 200;  // approximate visual height in pixels

  for (let i = 0; i < nPoles; i++) {
    const line = poleGroup.children[i];
    if (!line) continue;

    const off = (i * T + frame) * 2;
    const px = polePositionsData[off];
    const py = polePositionsData[off + 1];

    // NaN check — hide pole during occlusion gaps
    if (isNaN(px) || isNaN(py)) {
      line.visible = false;
      continue;
    }

    // Back-project 2D to approximate 3D at body depth
    const cx = (px - 960) * depth / focalLength;
    const cy = (py - 540) * depth / focalLength;

    // Vertical line in camera-space (flip Y/Z for Three.js)
    const baseY = -cy;
    const topY = baseY + (poleHeightPx * depth / focalLength);
    const posArr = line.geometry.attributes.position.array;
    posArr[0] = cx;  posArr[1] = baseY;  posArr[2] = -depth;
    posArr[3] = cx;  posArr[4] = topY;   posArr[5] = -depth;
    line.geometry.attributes.position.needsUpdate = true;
    line.visible = true;
  }
}

// ─── Camera Framing ──────────────────────────────────────────────────────────

let initialCameraPos = null;
let initialTarget = null;

function frameCameraOnBody() {
  if (isWorldSpace) {
    // World space: use camera position/target from metadata
    const cp = metadata.camera_position;
    const ct = metadata.camera_target;
    const cu = metadata.camera_up;
    camera.position.set(cp[0], cp[1], cp[2]);
    camera.up.set(cu[0], cu[1], cu[2]);
    controls.target.set(ct[0], ct[1], ct[2]);
    controls.update();
  } else {
    // Camera space: compute body center from first-frame vertices
    const V = metadata.vertex_count;
    let cx = 0, cy = 0, cz = 0;
    for (let i = 0; i < V; i++) {
      cx += verticesData[i * 3];
      cy += -verticesData[i * 3 + 1];
      cz += -verticesData[i * 3 + 2];
    }
    cx /= V;
    cy /= V;
    cz /= V;

    controls.target.set(cx, cy, cz);
    const bodyDist = Math.abs(cz);
    camera.position.set(0, 0, bodyDist * 0.4);
    controls.update();
  }

  // Save initial state for reset
  initialCameraPos = camera.position.clone();
  initialTarget = controls.target.clone();
}

function resetView() {
  if (initialCameraPos && initialTarget) {
    camera.position.copy(initialCameraPos);
    controls.target.copy(initialTarget);
    controls.update();
  }
}

// ─── Timeline Range (mobile: T0 to Release) ─────────────────────────────────

let lastHammerFrame = 0;  // cached: last non-NaN hammer frame
let activeRangePreset = 'all';

function computeLastHammerFrame() {
  const T = metadata.frame_count;
  lastHammerFrame = T - 1;
  if (hammerData) {
    for (let f = T - 1; f >= 0; f--) {
      const off = f * 3;
      const isInvalid = isNaN(hammerData[off]) ||
        (hammerData[off] === 0 && hammerData[off + 1] === 0 && hammerData[off + 2] === 0);
      if (!isInvalid) {
        lastHammerFrame = f;
        break;
      }
    }
  }
}

function getTimelineRange() {
  const tw = metadata.throw_window || {};
  const t0 = tw.start || 0;
  const totalFrames = metadata.frame_count - 1;

  if (activeRangePreset === 'wind') {
    return { min: 0, max: Math.max(t0, 0) };
  }
  if (activeRangePreset === 'throw') {
    const rel = tw.release || totalFrames;
    const fps = metadata.fps || 30;
    const postReleaseCap = Math.min(rel + Math.round(fps * 2), totalFrames);
    return { min: t0, max: postReleaseCap };
  }
  // 'all'
  return { min: 0, max: totalFrames };
}

function applyTimelineRange() {
  const range = getTimelineRange();
  timelineMin = range.min;
  timelineMax = range.max;
  const scrubber = document.getElementById('scrubber');
  if (scrubber) {
    scrubber.min = timelineMin;
    scrubber.max = timelineMax;
    if (currentFrame < timelineMin) { currentFrame = timelineMin; scrubber.value = currentFrame; updateFrame(currentFrame); }
    if (currentFrame > timelineMax) { currentFrame = timelineMax; scrubber.value = currentFrame; updateFrame(currentFrame); }
  }
  rebuildMarkers();
  positionThrowWindowBar();
  renderProvenanceBar();
}

function positionThrowWindowBar() {
  const bar = document.getElementById('throw-window-bar');
  if (!bar || !metadata || !metadata.throw_window) return;
  const range = timelineMax - timelineMin;
  if (range <= 0) return;
  const tw = metadata.throw_window;
  const start = Math.max(tw.start || 0, timelineMin);
  const release = Math.min(tw.release || timelineMax, timelineMax);
  const leftPct = ((start - timelineMin) / range) * 100;
  const widthPct = ((release - start) / range) * 100;
  bar.style.left = leftPct + '%';
  bar.style.width = widthPct + '%';
}

function renderProvenanceBar() {
  const canvas = document.getElementById('provenance-bar');
  if (!canvas || !hammerProvenanceData || metadata.hammer_source === 'optimized') {
    if (canvas) canvas.style.display = 'none';
    return;
  }
  canvas.style.display = 'block';
  const container = canvas.parentElement;
  const w = container.clientWidth;
  const h = 4;
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, w, h);

  const range = timelineMax - timelineMin;
  if (range <= 0) return;

  // Colors: 0=LIFTED green, 1=FALLBACK yellow, 2=SCREENED red, 3=NAN gray
  const colors = ['#4CAF50', '#FFC107', '#F44336', '#9E9E9E'];

  for (let f = timelineMin; f <= timelineMax; f++) {
    if (f < 0 || f >= hammerProvenanceData.length) continue;
    const prov = hammerProvenanceData[f];
    const color = colors[prov] || colors[3];
    const x = ((f - timelineMin) / range) * w;
    const nx = ((f + 1 - timelineMin) / range) * w;
    ctx.fillStyle = color;
    ctx.fillRect(Math.floor(x), 0, Math.ceil(nx - x), h);
  }
}

function rebuildMarkers() {
  const container = document.getElementById('markers');
  if (!container || !metadata) return;
  container.innerHTML = '';
  const range = timelineMax - timelineMin;
  if (range <= 0) return;

  if (metadata.turn_boundaries) {
    metadata.turn_boundaries.forEach((frame, i) => {
      if (frame < timelineMin || frame > timelineMax) return;
      const pct = ((frame - timelineMin) / range) * 100;
      const label = metadata.turn_labels ? metadata.turn_labels[i] : `T${i}`;
      const el = document.createElement('div');
      el.className = 'turn-marker';
      el.style.left = pct + '%';
      el.innerHTML = `<div class="tick"></div><div class="label">${label}</div>`;
      container.appendChild(el);
    });
  }

  const relFrame = metadata.throw_window && metadata.throw_window.release;
  if (relFrame && relFrame >= timelineMin && relFrame <= timelineMax) {
    const pct = ((relFrame - timelineMin) / range) * 100;
    const el = document.createElement('div');
    el.className = 'turn-marker release';
    el.style.left = pct + '%';
    el.innerHTML = `<div class="tick"></div><div class="label">REL</div>`;
    container.appendChild(el);
  }
}

// ─── UI ──────────────────────────────────────────────────────────────────────

// ─── Graph Overlay System ─────────────────────────────────────────────────────

let graphOverlayActive = false;
let activeOverlaySource = null;  // 'planes' or 'separation'

function showGraphOverlay(containerId, drawFn, source) {
  const isMobile = window.innerWidth <= 768 || window.innerHeight <= 500;
  const metricGraphs = document.getElementById('metric-graphs');
  const container = document.getElementById(containerId);
  if (!container) return;

  graphOverlayActive = true;
  activeOverlaySource = source || null;
  container.style.display = 'block';

  if (isMobile) {
    metricGraphs.classList.add('overlay-mode');
    requestAnimationFrame(() => {
      const canvas = container.querySelector('canvas');
      if (canvas) {
        const rect = canvas.getBoundingClientRect();
        canvas.width = Math.round(rect.width);
        canvas.height = Math.round(rect.height);
      }
      drawFn(currentFrame);
    });
  } else {
    drawFn(currentFrame);
  }
}

function dismissGraphOverlay() {
  const metricGraphs = document.getElementById('metric-graphs');
  graphOverlayActive = false;
  metricGraphs.classList.remove('overlay-mode');

  if (activeOverlaySource === 'planes') {
    const kaContainer = document.getElementById('kneeangle-container');
    if (kaContainer) kaContainer.style.display = 'none';
    const canvas = document.getElementById('kneeangle-graph');
    if (canvas) { canvas.width = 360; canvas.height = 140; }
    if (legPlanesGroup) legPlanesGroup.visible = false;
    const planesBtn = document.querySelector('.toggle-btn[data-target="planes"]');
    if (planesBtn) planesBtn.classList.remove('active');
  } else if (activeOverlaySource === 'separation') {
    const sepContainer = document.getElementById('separation-container');
    if (sepContainer) sepContainer.style.display = 'none';
    const canvas = document.getElementById('separation-graph');
    if (canvas) { canvas.width = 360; canvas.height = 140; }
    separationEnabled = false;
    updateTorsoColors(currentFrame);
    const sepBtn = document.querySelector('.toggle-btn[data-target="separation"]');
    if (sepBtn) sepBtn.classList.remove('active');
  }

  activeOverlaySource = null;
}

function cleanThrowName(raw) {
  if (metadata.display_name) return metadata.display_name;
  let name = raw || '';
  name = name.replace(/_\d{8,}$/g, '');       // strip date suffix like _20220626
  name = name.replace(/_small$/i, '');          // strip _small
  name = name.replace(/_/g, ' ');               // underscores to spaces
  return name || raw;
}

function initUI() {
  const T = metadata.frame_count;

  // Header — clean display name
  document.getElementById('throw-name').textContent = cleanThrowName(metadata.throw);

  // Lift quality badge
  const lqEl = document.getElementById('lift-quality');
  if (lqEl && metadata.lift_quality) {
    const lq = metadata.lift_quality;
    const label = lq.label || 'Unknown';
    const pct = Math.round(lq.lift_pct_throw || 0);
    lqEl.textContent = `Lift: ${pct}% (${label})`;
    lqEl.className = 'lift-quality ' + label.toLowerCase();
  }

  // Scrubber
  const scrubber = document.getElementById('scrubber');
  scrubber.max = T - 1;
  scrubber.value = 0;

  scrubber.addEventListener('input', (e) => {
    currentFrame = parseInt(e.target.value, 10);
    updateFrame(currentFrame);
  });

  // Play/pause
  const playBtn = document.getElementById('play-btn');
  playBtn.addEventListener('click', () => {
    playing = !playing;
    playBtn.innerHTML = playing ? '&#9646;&#9646;' : '&#9654;';
    if (playing) lastFrameTime = performance.now();
  });

  // Speed buttons
  document.querySelectorAll('.speed-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.speed-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      playbackSpeed = parseFloat(btn.dataset.speed);
    });
  });

  // Range preset buttons
  document.querySelectorAll('.range-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.range-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      activeRangePreset = btn.dataset.range;
      applyTimelineRange();
      // Redraw any visible graphs with new range
      const sepC = document.getElementById('separation-container');
      if (sepC && sepC.style.display !== 'none' && separationAngles) drawSeparationGraph(currentFrame);
      const btC = document.getElementById('backtilt-container');
      if (btC && btC.style.display !== 'none' && backTiltAngles) drawBackTiltGraph(currentFrame);
      const kaC = document.getElementById('kneeangle-container');
      if (kaC && kaC.style.display !== 'none' && legAlignmentData) drawLegCorotationGraph(currentFrame);
    });
  });

  // Keyboard shortcuts
  document.addEventListener('keydown', (e) => {
    if (e.code === 'Space') {
      e.preventDefault();
      playBtn.click();
    } else if (e.code === 'ArrowLeft') {
      e.preventDefault();
      setFrame(Math.max(timelineMin, currentFrame - 1));
    } else if (e.code === 'ArrowRight') {
      e.preventDefault();
      setFrame(Math.min(timelineMax, currentFrame + 1));
    } else if (e.code === 'Home') {
      e.preventDefault();
      setFrame(timelineMin);
    } else if (e.code === 'End') {
      e.preventDefault();
      setFrame(timelineMax);
    }
  });

  // View reset button
  document.getElementById('reset-view-btn').addEventListener('click', () => {
    resetView();
  });

  // Fullscreen toggle
  const fsBtn = document.getElementById('fullscreen-btn');
  if (fsBtn) {
    fsBtn.addEventListener('click', () => {
      if (!document.fullscreenElement) {
        document.documentElement.requestFullscreen().catch(() => {});
      } else {
        document.exitFullscreen().catch(() => {});
      }
    });
    document.addEventListener('fullscreenchange', () => {
      fsBtn.textContent = document.fullscreenElement ? '\u2716' : '\u26F6';
    });
  }

  // Screenshot button
  const ssBtn = document.getElementById('screenshot-btn');
  if (ssBtn) {
    ssBtn.addEventListener('click', () => {
      const dataURL = renderer.domElement.toDataURL('image/png');
      const link = document.createElement('a');
      const throwName = (metadata.throw || 'throw').replace(/\s+/g, '_');
      link.download = `${throwName}_frame${currentFrame}.png`;
      link.href = dataURL;
      link.click();
    });
  }

  // Hamburger menu toggle
  const hamburgerBtn = document.getElementById('hamburger-btn');
  const togglesPanel = document.getElementById('toggles');
  if (hamburgerBtn && togglesPanel) {
    hamburgerBtn.addEventListener('click', () => {
      togglesPanel.classList.toggle('open');
    });
    // Close menu when tapping outside (on the 3D canvas)
    document.getElementById('canvas-container').addEventListener('click', () => {
      togglesPanel.classList.remove('open');
    });
  }

  // Timeline range + turn markers
  applyTimelineRange();

  // Visibility toggles — Leg Planes uses tap/hold, others use simple click
  const planesBtn = document.querySelector('.toggle-btn[data-target="planes"]');
  if (planesBtn) {
    let holdTimer = null;
    let isHold = false;

    const onPointerDown = (e) => {
      e.preventDefault();
      isHold = false;
      holdTimer = setTimeout(() => {
        isHold = true;
        // Show leg planes at current frame (static)
        planesBtn.classList.add('active');
        legPlanesGroup.visible = true;
        updateOneLegPlane(currentFrame, leftLegPlane, KP_LEFT_HIP, KP_LEFT_KNEE, KP_LEFT_ANKLE);
        updateOneLegPlane(currentFrame, rightLegPlane, KP_RIGHT_HIP, KP_RIGHT_KNEE, KP_RIGHT_ANKLE);
        // After 1s flash, show graph overlay
        setTimeout(() => {
          showGraphOverlay('kneeangle-container', drawLegCorotationGraph, 'planes');
        }, 1000);
      }, 500);
    };

    const onPointerUp = (e) => {
      clearTimeout(holdTimer);
      if (!isHold) {
        // Tap: toggle 3D planes only (no graph)
        if (graphOverlayActive) {
          // If graph overlay is showing, dismiss it
          dismissGraphOverlay();
        } else {
          // Toggle planes
          planesBtn.classList.toggle('active');
          const visible = planesBtn.classList.contains('active');
          legPlanesGroup.visible = visible;
          if (visible) {
            updateOneLegPlane(currentFrame, leftLegPlane, KP_LEFT_HIP, KP_LEFT_KNEE, KP_LEFT_ANKLE);
            updateOneLegPlane(currentFrame, rightLegPlane, KP_RIGHT_HIP, KP_RIGHT_KNEE, KP_RIGHT_ANKLE);
          }
        }
      }
    };

    planesBtn.addEventListener('pointerdown', onPointerDown);
    planesBtn.addEventListener('pointerup', onPointerUp);
    planesBtn.addEventListener('pointercancel', () => clearTimeout(holdTimer));
    planesBtn.addEventListener('contextmenu', (e) => e.preventDefault());
    planesBtn.addEventListener('selectstart', (e) => e.preventDefault());
    planesBtn.addEventListener('click', (e) => { e.preventDefault(); e.stopPropagation(); });
  }

  // Separation button — tap/hold like Leg Planes (mobile)
  const sepBtn = document.querySelector('.toggle-btn[data-target="separation"]');
  if (sepBtn) {
    let sepHoldTimer = null;
    let sepIsHold = false;

    const onSepDown = (e) => {
      e.preventDefault();
      sepIsHold = false;
      sepHoldTimer = setTimeout(() => {
        sepIsHold = true;
        sepBtn.classList.add('active');
        separationEnabled = true;
        updateTorsoColors(currentFrame);
        setTimeout(() => {
          showGraphOverlay('separation-container', drawSeparationGraph, 'separation');
        }, 1000);
      }, 500);
    };

    const onSepUp = (e) => {
      clearTimeout(sepHoldTimer);
      if (!sepIsHold) {
        if (graphOverlayActive && activeOverlaySource === 'separation') {
          dismissGraphOverlay();
        } else {
          sepBtn.classList.toggle('active');
          const visible = sepBtn.classList.contains('active');
          separationEnabled = visible;
          updateTorsoColors(currentFrame);
        }
      }
    };

    sepBtn.addEventListener('pointerdown', onSepDown);
    sepBtn.addEventListener('pointerup', onSepUp);
    sepBtn.addEventListener('pointercancel', () => clearTimeout(sepHoldTimer));
    sepBtn.addEventListener('contextmenu', (e) => e.preventDefault());
    sepBtn.addEventListener('selectstart', (e) => e.preventDefault());
    sepBtn.addEventListener('click', (e) => { e.preventDefault(); e.stopPropagation(); });
  }

  // Tap on graph container dismisses overlay
  const kneeContainer = document.getElementById('kneeangle-container');
  if (kneeContainer) {
    kneeContainer.addEventListener('click', () => {
      if (graphOverlayActive) dismissGraphOverlay();
    });
  }
  const sepContainer = document.getElementById('separation-container');
  if (sepContainer) {
    sepContainer.addEventListener('click', () => {
      if (graphOverlayActive) dismissGraphOverlay();
    });
  }

  // Other visibility toggles (backtilt, circle, maxmin)
  document.querySelectorAll('.toggle-btn').forEach(btn => {
    const target = btn.dataset.target;
    if (target === 'planes' || target === 'separation') return;  // handled above
    btn.addEventListener('click', () => {
      btn.classList.toggle('active');
      const visible = btn.classList.contains('active');
      if (target === 'backtilt') {
        backPlanesGroup.visible = visible;
        const btContainer = document.getElementById('backtilt-container');
        if (btContainer) btContainer.style.display = visible ? 'block' : 'none';
        if (visible) updateBackPlane(currentFrame);
      }
      if (target === 'maxmin' && orbitExtremesGroup) {
        orbitExtremesGroup.visible = visible;
        if (visible) updateOrbitExtremesFrame(currentFrame);
      }
      if (target === 'circle' && circleGroup) circleGroup.visible = visible;
    });
  });

  // Model selector — swap between Sean's raw mesh and rigged GLBs
  const modelSelect = document.getElementById('model-select');
  if (modelSelect) {
    modelSelect.addEventListener('change', async () => {
      const val = modelSelect.value;
      if (val === 'sean_mesh') {
        // Switch to raw mesh mode
        if (riggedRoot) { scene.remove(riggedRoot); riggedRoot = null; }
        riggedModel = null;
        riggedSkeleton = null;
        riggedBones = {};
        isRigged = false;
        if (!bodyMesh && verticesData && facesData) createBodyMesh();
        if (bodyMesh) bodyMesh.visible = true;
        updateFrame(currentFrame);
      } else {
        // Switch to rigged GLB
        if (bodyMesh) bodyMesh.visible = false;
        isRigged = true;
        const glbUrl = `models/${val}.glb`;
        try {
          await loadRiggedModel(glbUrl);
          updateFrame(currentFrame);
        } catch (err) {
          console.error('Model swap failed:', err);
          modelSelect.value = 'humanoid';
          await loadRiggedModel('models/humanoid.glb');
          updateFrame(currentFrame);
        }
      }
    });
  }
}


function setFrame(f) {
  currentFrame = f;
  document.getElementById('scrubber').value = f;
  updateFrame(f);
}

function updateFrame(frame) {
  if (isRigged) {
    updateRiggedFrame(frame);
  } else {
    updateMeshFrame(frame);
  }
  updateHammerFrame(frame);
  updateOrbitExtremesFrame(frame);
  updateLegPlanes(frame);
  updateBackPlane(frame);
  updateTorsoColors(frame);
  updateCircleFrame(frame);
  updateFrameDisplay(frame);

  // Update separation graph if visible
  const sepContainer = document.getElementById('separation-container');
  if (sepContainer && sepContainer.style.display !== 'none' && separationAngles) {
    drawSeparationGraph(frame);
  }
}

function updateFrameDisplay(frame) {
  const T = metadata.frame_count;
  const timeS = (frame / metadata.fps).toFixed(2);

  // Wire length suffix
  let wireStr = '';
  if (wireDistanceData && frame < wireDistanceData.length) {
    const w = wireDistanceData[frame];
    if (isFinite(w) && w > 0) wireStr = `  W=${w.toFixed(2)}m`;
  }

  document.getElementById('frame-info').textContent =
    `Frame ${frame} / ${T - 1}  (${timeS}s)${wireStr}`;

  // Update PiP video frame from preloaded thumbnails
  const pipFrame = document.getElementById('pip-frame');
  if (pipFrame && pipFrames && pipFrames[frame] && pipFrames[frame].complete) {
    pipFrame.src = pipFrames[frame].src;
    const pipLabel = document.getElementById('pip-label');
    if (pipLabel) pipLabel.textContent = `Frame ${frame} (${timeS}s)`;
  }

  // Draw 2D reprojection dots on PiP overlay
  const pipOverlay = document.getElementById('pip-overlay');
  if (pipOverlay) {
    const pipImg = document.getElementById('pip-frame');
    const rect = pipImg.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    pipOverlay.width = rect.width * dpr;
    pipOverlay.height = rect.height * dpr;
    const ctx = pipOverlay.getContext('2d');
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, rect.width, rect.height);

    const sx = rect.width / 1920, sy = rect.height / 1080;
    const off = frame * 2;

    // Green = original 2D detection
    if (hammerOrig2d && off + 1 < hammerOrig2d.length) {
      const ox = hammerOrig2d[off], oy = hammerOrig2d[off + 1];
      if (isFinite(ox) && isFinite(oy)) {
        ctx.beginPath(); ctx.arc(ox * sx, oy * sy, 4, 0, 2 * Math.PI);
        ctx.fillStyle = '#00ff00'; ctx.fill();
      }
    }
    // Orange = reprojected 3D
    if (hammerReproj2d && off + 1 < hammerReproj2d.length) {
      const rx = hammerReproj2d[off], ry = hammerReproj2d[off + 1];
      if (isFinite(rx) && isFinite(ry)) {
        ctx.beginPath(); ctx.arc(rx * sx, ry * sy, 4, 0, 2 * Math.PI);
        ctx.fillStyle = '#ff8c00'; ctx.fill();
      }
    }
  }
}

// ─── Animation Loop ──────────────────────────────────────────────────────────

function animate() {
  requestAnimationFrame(animate);

  if (playing) {
    const now = performance.now();
    const elapsed = now - lastFrameTime;
    const frameDuration = 1000 / (metadata.fps * playbackSpeed);

    if (elapsed >= frameDuration) {
      lastFrameTime = now - (elapsed % frameDuration);
      currentFrame++;
      if (currentFrame > timelineMax) {
        currentFrame = timelineMin;
      }
      document.getElementById('scrubber').value = currentFrame;
      updateFrame(currentFrame);
    }
  }

  controls.update();
  renderer.render(scene, camera);
}

// ─── Color Legend ─────────────────────────────────────────────────────────────

function createColorLegend() {
  if (!fillTypeData) return;
  const legend = document.getElementById('color-legend');
  if (!legend) return;

  // Count occurrences of each fill_type
  const counts = [0, 0, 0, 0, 0];
  for (let i = 0; i < fillTypeData.length; i++) {
    const v = fillTypeData[i];
    if (v >= 0 && v <= 4) counts[v]++;
  }

  // Only show types that actually appear
  for (let i = 0; i < FILL_TYPE_COLORS.length; i++) {
    if (counts[i] === 0) continue;
    const item = document.createElement('div');
    item.className = 'legend-item';
    item.innerHTML = `<span class="legend-dot" style="background:${FILL_TYPE_COLORS[i].css}"></span>${FILL_TYPE_COLORS[i].label} <span class="legend-count">${counts[i]}</span>`;
    legend.appendChild(item);
  }

  // Sim overlay legend entries
  if (simHammerFwdData && metadata.sim_overlay) {
    const fwdMeta = metadata.sim_overlay.fwd || metadata.sim_overlay;
    const fwdItem = document.createElement('div');
    fwdItem.className = 'legend-item';
    const fwdWire = fwdMeta.wire_length ? ` (L=${fwdMeta.wire_length.toFixed(2)}m)` : '';
    fwdItem.innerHTML = `<span class="legend-dot" style="background:#00ff88"></span>Sim Forward${fwdWire}`;
    legend.appendChild(fwdItem);
  }
  if (simHammerBwdData && metadata.sim_overlay?.bwd) {
    const bwdMeta = metadata.sim_overlay.bwd;
    const bwdItem = document.createElement('div');
    bwdItem.className = 'legend-item';
    const bwdWire = bwdMeta.wire_length ? ` (L=${bwdMeta.wire_length.toFixed(2)}m)` : '';
    bwdItem.innerHTML = `<span class="legend-dot" style="background:#4488ff"></span>Sim Backward${bwdWire}`;
    legend.appendChild(bwdItem);
  }
}

// ─── Main ────────────────────────────────────────────────────────────────────

async function main() {
  try {
    await loadData();
  } catch (err) {
    document.getElementById('loading').innerHTML =
      `<div class="label" style="color:#ff6b6b;">Failed to load data: ${err.message}</div>`;
    return;
  }

  computeLastHammerFrame();
  initScene();

  // Default to Sean's raw mesh (user sees themselves first)
  // Rigged GLB avatars are loaded on demand via model selector
  if (verticesData && facesData) {
    createBodyMesh();
    isRigged = false;  // start in raw mesh mode
  } else if (isRigged) {
    try {
      await loadRiggedModel();
      console.log('Rigged model ready');
    } catch (err) {
      console.error('Failed to load rigged model:', err);
    }
  }

  createHammer();
  createGround();
  createLegPlanes();
  createBackPlane();
  precomputeSeparation();
  createCircle();
  createCircleLabels();
  setupLabelDrag();
  precomputeOrbitExtremes();
  createOrbitExtremesSpheres();

  // Set initial frame — positionGround first so groundY is available
  positionGround(0);

  // Snap circle Y to ground level (camera space only)
  if (!isWorldSpace && circleLine && groundY > -Infinity) {
    circleLine.position.y = groundY;
    if (circleLabelsGroup) circleLabelsGroup.position.y = groundY;
  }

  updateFrame(0);
  frameCameraOnBody();

  initUI();
  createColorLegend();

  // Hide loading
  document.getElementById('loading').classList.add('hidden');

  // Show onboarding on first visit
  if (!localStorage.getItem('throwsage_onboarded')) {
    const onboarding = document.getElementById('onboarding');
    if (onboarding) {
      onboarding.classList.remove('hidden');
      const dismiss = () => {
        onboarding.classList.add('hidden');
        localStorage.setItem('throwsage_onboarded', '1');
      };
      onboarding.addEventListener('click', dismiss);
      setTimeout(dismiss, 6000);
    }
  }

  // Start render loop
  animate();
}

main();
