// pages/api/save-chat.js
import dbConnect from "../../lib/dbConnect";
import Chat from "../../models/Chat";

export default async function handler(req, res) {
  await dbConnect();

  if (req.method === "POST") {
    try {
      const { query, response, email } = req.body;
      // Create a new chat document
      const newChat = new Chat({ query, response, email });
      await newChat.save();
      return res.status(201).json({ message: "Chat saved successfully", chat: newChat });
    } catch (error) {
      console.error("Error saving chat:", error);
      return res.status(500).json({ message: "Internal Server Error" });
    }
  } else {
    res.setHeader("Allow", ["POST"]);
    return res.status(405).json({ message: `Method ${req.method} Not Allowed` });
  }
}
