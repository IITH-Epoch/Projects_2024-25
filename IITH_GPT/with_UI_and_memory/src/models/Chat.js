import mongoose from "mongoose";

const ChatSchema = new mongoose.Schema({
  query: { type: String, required: true },
  response: { type: String, required: true },
  email: { type: String, required: true },
  timestamp: { type: Date, default: Date.now },
});

export default mongoose.models.Chat || mongoose.model("Chat", ChatSchema);
